# from mpi4py import MPI
# from GKTServerTrainer import GKTServerTrainer
# from GKTClientTrainer import GKTClientTrainer
import argparse
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List, Any

import torchvision
from torch._C._monitor import MEAN
from torch.nn import functional as F
import copy
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, EMNIST
from torchvision.transforms import Compose, Normalize

import wandb
import utils
import torch
from torch import nn, Tensor

from data_util.FMNIST.fashionmnist_data_loader import _data_transforms_FashionMNIST
from data_util.constants import MEAN, STD

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

PROJECT_DIR = Path(__file__).absolute().parent


def get_fedmd_argparser(args) -> ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.set_defaults(**vars(args))
    parser.add_argument("--digest_epoch", type=int, default=200)
    parser.add_argument("--local_epoch", type=int, default=3)
    parser.add_argument("--public_epoch", type=int, default=200)
    return parser


class FedMD_standalone_API:
    def __init__(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                 train_data_local_dict, test_data_local_dict, args, test_data_global):
        self.client_models = client_models
        self.test_data_global = test_data_global
        self.criterion_KL = utils.KL_Loss(args.temperature)
        self.criterion_CE = F.cross_entropy
        self.args = get_fedmd_argparser(args).parse_args()
        self.consensus: List[torch.Tensor] = []
        self.mse_criterion = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.public_batch_class = 10

    def custom_cross_entropy(self,raw_output, true_labels):
        # 应用 LogSoftmax 激活
        log_softmax_output = F.log_softmax(raw_output, dim=1)

        # 计算每个样本的交叉熵损失
        loss_per_sample = -torch.sum(true_labels * log_softmax_output, dim=1)

        # 计算平均损失
        average_loss = torch.mean(loss_per_sample)

        return average_loss

    def aggregate(self, scores_cache: List[torch.Tensor]) -> torch.Tensor:
        total_sum = torch.zeros_like(scores_cache[0])
        # 遍历 scores_cache，将每个张量相加到 total_sum
        for tensor in scores_cache:
            total_sum += tensor
        # 求平均值
        average = total_sum / len(scores_cache)
        consensus = average
        return consensus


    # 处理每个客户的初始数据 end

    def do_fedMD_stand_alone(self, client_models, train_data_local_num_dict, test_data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args, public_train_data, public_test_data):
        wandb.login(key="8eece390c9549c98f5adc1b49b53b38a5c4ebb74")
        wandb.init(project='FedMD', config=args)


        # 开始
        scores_cache: list[Tensor] = []

        # torch.autograd.set_detect_anomaly(True)

        # 每个客户训练公共数据
        for client_index, model in enumerate(self.client_models):
            # compute on public
            print("开始公共训练第" + str(client_index) + "个客户端")
            model.to(self.device)
            model.train()
            public_train_loss_avg = utils.RunningAverage()
            public_train_accTop1_avg = utils.RunningAverage()
            public_train_accTop5_avg = utils.RunningAverage()
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=args.wd)
            for _ in range(self.args.public_epoch):
                for i in range(len(public_train_data)):
                    # print("第"+str(client_index)+ "的第" + str(i) + "轮公共训练")
                    for batch_idx, (images, labels) in enumerate(public_train_data[i]):
                        images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
                        log_probs = model(images)
                        loss = self.criterion_CE(log_probs, labels)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

            for _ in range(self.args.public_epoch):
                for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                    labels = torch.tensor(labels, dtype=torch.long)
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_probs = model(images)
                    loss = self.criterion_CE(log_probs, labels)

                    # Update average loss and accuracy
                    public_train_metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                    # only one element tensors can be converted to Python scalars
                    public_train_accTop1_avg.update(public_train_metrics[0].item())
                    public_train_accTop5_avg.update(public_train_metrics[1].item())
                    public_train_loss_avg.update(loss.item())

                    wandb.log({f"public top1 train Model {client_index} Accuracy": public_train_accTop1_avg.value()})
                    wandb.log({f"public top5 train Model {client_index} Accuracy": public_train_accTop5_avg.value()})
                    wandb.log({f"public loss train Model {client_index} Accuracy": public_train_loss_avg.value()})

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

        # 验证敛散性
        public_acc_all = []
        for client_index, client_model in enumerate(self.client_models):
            # 验证客户端的准确性
            print("开始验证第" + str(client_index) + "个客户端的敛散性")
            client_model.eval()
            public_test_loss_avg = utils.RunningAverage()
            public_test_accTop1_avg = utils.RunningAverage()
            public_test_accTop5_avg = utils.RunningAverage()
            for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long)
                log_probs = client_model(images)
                loss = self.criterion_CE(log_probs, labels)
                # Update average loss and accuracy
                pubic_test_metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                # only one element tensors can be converted to Python scalars
                public_test_accTop1_avg.update(pubic_test_metrics[0].item())
                public_test_accTop5_avg.update(pubic_test_metrics[1].item())
                public_test_loss_avg.update(loss.item())
                wandb.log({f"public top1 test Model {client_index} Accuracy": public_test_accTop1_avg.value()})
                wandb.log({f"public top5 test Model {client_index} Accuracy": public_test_accTop5_avg.value()})
                wandb.log({f"public loss test Model {client_index} Accuracy": public_test_loss_avg.value()})
            # print(loss_avg,type(loss_avg))

            public_acc_all.append(public_test_accTop1_avg.value())
        wandb.log({"local mean Test/AccTop1": float(np.mean(np.array(public_acc_all)))})

        #上传score
        digest_loss_avg = utils.RunningAverage()
        digest_accTop1_avg = utils.RunningAverage()
        digest_accTop5_avg = utils.RunningAverage()
        for _ in range(self.args.digest_epoch):
            for i in range(len(public_test_data)):
                for batch_idx, (images, labels) in enumerate(public_train_data[i]):
                    scores_cache = []
                    for client_index, model in enumerate(self.client_models):
                        #监控数据
                        model.eval()

                        with torch.no_grad():
                            images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
                            log_probs = model(images)
                            changed_score = F.softmax(log_probs)
                            # 选择一个破坏者
                            # 选择一个破坏者【在这里进行攻击！！！】
                            # if client_index < 3:
                            #     changed_score = utils.change_logits(changed_score)
                            #     log_probs = utils.repalceLogitsWith0(log_probs)
                            # log_probs = utils.replace_logits_with_random(log_probs)
                            scores_cache.append(changed_score.detach())



                    # aggregate
                    self.consensus = self.aggregate(scores_cache)

                    # digest
                    for client_index, model in enumerate(self.client_models):
                        # 开始digest
                        model.train()
                        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                                weight_decay=args.wd)
                        # for batch_idx, (images, labels) in enumerate(public_train_data[0]):
                        #     labels = torch.tensor(labels, dtype=torch.long)
                        #     images, labels = images.to(self.device), labels.to(self.device)
                        log_probs = model(images)
                        loss = self.custom_cross_entropy(log_probs, self.consensus)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                # revisit
                for _ in range(self.args.local_epoch):
                    for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                        labels = torch.tensor(labels, dtype=torch.long)
                        images, labels = images.to(self.device), labels.to(self.device)
                        log_probs = model(images)
                        loss = self.criterion_CE(log_probs, labels)

                        # Update average loss and accuracy
                        metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                        # only one element tensors can be converted to Python scalars
                        digest_accTop1_avg.update(metrics[0].item())
                        digest_accTop5_avg.update(metrics[1].item())
                        digest_loss_avg.update(loss.item())
                        wandb.log({f"digest top1 test Model {client_index} Accuracy": digest_accTop1_avg.value()})
                        wandb.log({f"digest top5 test Model {client_index} Accuracy": digest_accTop5_avg.value()})
                        wandb.log({f"digest loss test Model {client_index} Accuracy": digest_loss_avg.value()})

                        optim.zero_grad()
                        loss.backward()
                        optim.step()


        # 验证客户端的准确性
        acc_all = []
        for client_index, client_model in enumerate(self.client_models):
            # 验证客户端的准确性
            print("开始验证第" + str(client_index) + "个客户端")
            client_model.eval()
            loss_avg = utils.RunningAverage()
            accTop1_avg = utils.RunningAverage()
            accTop5_avg = utils.RunningAverage()
            for batch_idx, (images, labels) in enumerate(test_data_local_dict[client_index]):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long)
                log_probs = client_model(images)
                loss = self.criterion_CE(log_probs, labels)
                # Update average loss and accuracy
                metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                # only one element tensors can be converted to Python scalars
                accTop1_avg.update(metrics[0].item())
                accTop5_avg.update(metrics[1].item())
                loss_avg.update(loss.item())
                wandb.log({f"local top1 test Model {client_index} Accuracy": accTop1_avg.value()})
                wandb.log({f"local top5 test Model {client_index} Accuracy": accTop5_avg.value()})
                wandb.log({f"local loss test Model {client_index} Accuracy": loss_avg.value()})

            acc_all.append(accTop1_avg.value())
        wandb.log({"local mean Test/AccTop1": float(np.mean(np.array(acc_all)))})

        wandb.finish()
