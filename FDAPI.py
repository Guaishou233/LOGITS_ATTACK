# from mpi4py import MPI
# from GKTServerTrainer import GKTServerTrainer
# from GKTClientTrainer import GKTClientTrainer
from torch.nn import functional as F
import copy
import torch
import os
import numpy as np
import wandb
import utils
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from heapq import heappop as pop
from heapq import heappush as push

import copy
import queue

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from heapq import heappop as pop
from heapq import heappush as push
import copy
import queue


def tensor_cross_entropy(output: torch.Tensor, target: torch.Tensor):
    return -1.0 * (output.log() * target).mean()


np.random.seed(0)


def knowledge_avg(knowledge, weights):
    result = []
    for k_ in knowledge:
        result.append(knowledge_avg_single(k_, weights))
    return torch.Tensor(np.array(result)).cuda()


def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0].knowledge)
    sum = 0
    for _k, _w in zip(knowledge, weights):
        result = result + _k.knowledge * _w
        sum = sum + _w
    result = result / sum
    return np.array(result.detach().cpu())


class FD_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.temperature)
        self.criterion_CE = F.cross_entropy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def do_fd_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                          train_data_local_dict, test_data_local_dict, args):
        wandb.login(key="8eece390c9549c98f5adc1b49b53b38a5c4ebb74")
        wandb.init(project='FD', config=args)
        # 第一步 初始化全局知识
        global_knowledge = {}  # 输入类别 输出对应的知识
        local_knowledge = {}  # 输入客户端 输出类别->知识的字典
        # 初始化全局与局部的知识 【这里的知识其实就是模型的软目标】
        for c in range(args.class_num):
            global_knowledge[c] = torch.Tensor(
                np.array([1 / args.class_num for _ in range(args.class_num)])) * args.client_number
        for idx in range(len(self.client_models)):
            local_knowledge[idx] = {}
            for c in range(args.class_num):
                local_knowledge[idx][c] = torch.Tensor(np.array([1 / args.class_num for _ in range(args.class_num)]))

        for global_epoch in range(args.comm_round):  # 表示进行多少次客户端与服务器之间的交互，设置为无限大则一直不停
            metrics_all = {'test_loss': [], 'test_accTop1': [], 'test_accTop5': [], 'f1': []}
            for client_index, client_model in enumerate(self.client_models):
                tmp_logits = {}  # 第c类的所有logits【收集每个设备上的对应类别样本的输出，并且用二维数组对应类别存储[类别][输出]】
                for c in range(args.class_num):
                    tmp_logits[c] = []
                # 初始化用于每一个客户端的知识 全局给定标签
                # 本质上也是根据标签进行block知识 每一个标签对应的知识都不一样
                # FD方法并没有根据样本数量进行加权
                print("开始训练第" + str(client_index) + "个客户端")
                client_model = client_model.to(self.device)
                client_model.train()
                optim = torch.optim.SGD(client_model.parameters(), lr=args.lr, momentum=0.9,
                                        weight_decay=args.wd)
                # print("client",client_index)
                for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                    labels = torch.tensor(labels, dtype=torch.long)
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_probs = client_model(images)
                    # #选择一个破坏者【在这里进行攻击！！！】
                    if client_index < 1 :
                        log_probs = utils.change_logits(log_probs)
                        # log_probs = utils.repalceLogitsWith0(log_probs)
                    #
                        # log_probs = utils.replace_logits_with_random(log_probs)


                    loss_true = F.cross_entropy(log_probs, labels)
                    # 接下来挨个生成soft_label，并添加进入local_knowledge
                    soft_label = []
                    for logit, label in zip(log_probs, labels):
                        c = int(label)
                        soft_label.append(
                            (global_knowledge[c] - local_knowledge[client_index][c]) / (args.client_number - 1))
                        tmp_logits[c].append(logit.cpu().detach().numpy())
                    soft_label = torch.Tensor([item.cpu().detach().numpy() for item in soft_label]).to(self.device)
                    loss_kd = F.cross_entropy(log_probs, F.softmax(
                        soft_label))  # tensor_cross_entropy(log_probs,soft_label)#self.criterion_KL(log_probs, soft_label)
                    loss = loss_true + args.alpha * loss_kd
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    torch.cuda.empty_cache()
                # 处理tmp_logits
                for c in range(args.class_num):
                    if len(tmp_logits[c]) != 0:
                        local_knowledge[client_index][c] = torch.mean(torch.Tensor(np.array(tmp_logits[c])), 0)
                    else:
                        local_knowledge[client_index][c] = torch.Tensor(
                            np.array([1 / args.class_num for _ in range(args.class_num)]))

            # 处理global_logits
            for c in range(args.class_num):
                tmp = []
                for client_index in range(args.client_number):
                    # print(str(client_index),local_knowledge[client_index])
                    tmp.append(np.array(local_knowledge[client_index][c]))
                # print("class",c,":")
                # print("tmp are as follows:")
                # for ele in tmp:
                #    print(ele)
                global_knowledge[c] = torch.sum(torch.Tensor(np.array(tmp)).float(), 0)

            # print("#####global knowledge#####")
            for c in range(args.class_num):
                print(str(c) + ":", global_knowledge[c] / args.client_number)

            if global_epoch % args.interval == 0:
                acc_all = []
                for client_index, client_model in enumerate(self.client_models):
                    if client_index % args.sel != 0:
                        continue
                        # 验证客户端的准确性
                    print("开始验证第" + str(client_index) + "个客户端")
                    client_model.eval()
                    loss_avg = utils.RunningAverage()
                    accTop1_avg = utils.RunningAverage()
                    # accTop5_avg = utils.RunningAverage()
                    for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels = torch.tensor(labels, dtype=torch.long)
                        log_probs = client_model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        # Update average loss and accuracy
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        # only one element tensors can be converted to Python scalars
                        accTop1_avg.update(metrics[0].item())
                        # accTop5_avg.update(metrics[1].item())
                        wandb.log({f"local top1 test Model {client_index} Accuracy": accTop1_avg.value()})
                        loss_avg.update(loss.item())
                    # print(loss_avg,type(loss_avg))

                    # compute mean of all metrics in summary
                    test_metrics = {str(client_index) + ' test_loss': loss_avg.value(),
                                    str(client_index) + ' test_accTop1': accTop1_avg.value(),
                                    # str(client_index) + ' test_accTop5': accTop5_avg.value(),
                                    }
                    # wandb.log({str(client_index)+" Test/Loss": test_metrics[str(client_index)+' test_loss']})
                    wandb.log({str(client_index)+" Test/AccTop1": test_metrics[str(client_index)+' test_accTop1']})
                    acc_all.append(accTop1_avg.value())
                wandb.log({"mean Test/AccTop1": float(np.mean(np.array(acc_all)))})
                # metrics=self.eval_on_the_client()

                # for batch_idx, (images, labels) in enumerate(self.local_training_data):
                #    images, labels = images.to(self.device), labels.to(self.device)
                #    log_probs, extracted_features = self.client_model(images)
                #    extracted_feature_dict[batch_idx] = extracted_features.cpu().detach().numpy()
                #    log_probs = log_probs.cpu().detach().numpy()
                #    logits_dict[batch_idx] = log_probs
                #    labels_dict[batch_idx] = labels.cpu().detach().numpy()

                # for batch_idx, (images, labels) in enumerate(self.local_test_data):
                #    test_images, test_labels = images.to(self.device), labels.to(self.device)
                #    _, extracted_features_test = self.client_model(test_images)
                #    extracted_feature_dict_test[batch_idx] = extracted_features_test.cpu().detach().numpy()
                #    labels_dict_test[batch_idx] = test_labels.cpu().detach().numpy()

        wandb.finish()

