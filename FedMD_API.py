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
    parser.add_argument("--digest_epoch", type=int, default=5)
    parser.add_argument("--local_epoch", type=int, default=5)
    parser.add_argument("--public_epoch", type=int, default=5)
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

    def _data_transforms_FEMNIST(self):
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将通道数从1变为3
            transforms.ToTensor(),
            transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
        ])

        valid_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将通道数从1变为3
            transforms.ToTensor(),
            transforms.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
        ])

        return train_transform, valid_transform



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
        wandb.login(key="7b2c2faf25f89695e0818127528f37c246743c86")
        wandb.init(project='FedMD', config=args)


        # 开始
        scores_cache: list[Tensor] = []
        # 获取当前工作目录
        target_dir = '/model_for_fd/'
        path = str(PROJECT_DIR) + target_dir
        torch.autograd.set_detect_anomaly(True)

        # 每个客户训练公共数据
        for client_index, model in enumerate(self.client_models):
            # compute on public
            # 尝试加载模型参数
            model_params_file = path + f'{args.dataset}+{client_index}.pth'
            if os.path.exists(model_params_file):
                # 如果有保存的参数，初始化参数并保存
                model_params = torch.load(model_params_file)
                model.load_state_dict(model_params)
            else:
                # 如果不存在保存的参数，加载并覆盖初始参数
                model_params = model.state_dict()
                torch.save(model_params, model_params_file)
            print("开始公共训练第" + str(client_index) + "个客户端")
            model.to(self.device)
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.wd)
            for _ in range(self.args.public_epoch):
                for batch_idx, (images, labels) in enumerate(public_train_data[0]):

                    images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
                    log_probs = model(images)
                    loss = self.criterion_CE(log_probs, labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            client_params = model.state_dict()
            torch.save(client_params,model_params_file)
        #测试公共数据精度
        public_acc_all = []
        for client_index, model in enumerate(self.client_models):
            #监控数据
            model.to(self.device)
            model.eval()
            public_accTop1_avg = utils.RunningAverage()
            public_accTop5_avg = utils.RunningAverage()

            for batch_idx, (images, labels) in enumerate(public_test_data[0]):
                images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
                log_probs = model(images)
                # Update average loss and accuracy
                metrics = utils.accuracy(log_probs, labels, topk=(1, 5))

                # #选择一个破坏者
                if client_index == 0:
                    log_probs = utils.change_logits(log_probs)

                #
                #     num_columns = log_probs.size(1)
                #     # 使用 torch.topk 获取每一行的值和索引
                #     values, indices = torch.topk(log_probs, k=num_columns, dim=1)
                #     # print("-----values-------indices-------")
                #     # print(values)
                #     # print(indices)
                #     changed_indices = None
                #     for i in range(0, num_columns, 2):
                #         # 交换索引的位置
                #         swapped_indices = torch.cat([indices[:, i + 1:i + 2], indices[:, i:i + 1]], dim=1)
                #         # print("-----swapped_indices--------------")
                #         # print(swapped_indices)
                #         if i == 0:
                #             changed_indices = swapped_indices
                #         else:
                #             changed_indices = torch.cat([changed_indices,swapped_indices],dim=1)
                #         # print("-----changed_indices--------------")
                #         # print(changed_indices)
                #         # 更新 log_probs 中的索引
                #     values = values.to(log_probs)
                #     log_probs = log_probs.scatter(dim=1, index=changed_indices, src=values)
                #     # print("-----changed_log_probs--------------")
                #     # print(log_probs)

                scores_cache.append(log_probs)

                # only one element tensors can be converted to Python scalars
                public_accTop1_avg.update(metrics[0].item())
                public_accTop5_avg.update(metrics[1].item())
                wandb.log({f"public top1 test Model {client_index} Accuracy": public_accTop1_avg.value()})
                # wandb.log({f"public loss test Model {client_index} Accuracy": public_loss_avg.value()})
            public_acc_all.append(public_accTop1_avg.value())
        wandb.log({"public mean Test/AccTop1": float(np.mean(np.array(public_acc_all)))})

        # aggregate
        self.consensus = self.aggregate(scores_cache).detach()
        print(self.consensus)

        # digest & revisit
        for client_index, model in enumerate(self.client_models):
            model_params_file = path + f'{args.dataset}+{client_index}.pth'
            client_params = torch.load(model_params_file)
            # 开始本地训练
            print("开始本地训练第" + str(client_index) + "个客户端")
            model.to(self.device)
            model.load_state_dict(client_params)
            model.train()
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.wd)
            # digest
            for _ in range(self.args.digest_epoch):
                for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                    labels = torch.tensor(labels, dtype=torch.long)
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_probs = model(images)
                    loss = self.mse_criterion(log_probs, self.consensus)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            client_params = model.state_dict()
            torch.save(client_params, model_params_file)

            # revisit
            for _ in range(self.args.local_epoch):
                for batch_idx, (images, labels) in enumerate(train_data_local_dict[client_index]):
                    labels = torch.tensor(labels, dtype=torch.long)
                    images, labels = images.to(self.device), labels.to(self.device)
                    log_probs = model(images)
                    loss = self.criterion_CE(log_probs, labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            client_params = torch.load(model_params_file)
            torch.save(client_params, model_params_file)

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
                # wandb.log({f"local loss test Model {client_index} Accuracy": loss_avg.value()})
                # wandb.log({"client": client_index,"test_loss": loss_avg.value(),"test_accTop1": accTop1_avg.value(),"test_accTop5": accTop5_avg.value()})
            # print(loss_avg,type(loss_avg))

            # compute mean of all metrics in summary
            test_metrics = {str(client_index) + ' test_loss': loss_avg.value(),
                            str(client_index) + ' test_accTop1': accTop1_avg.value(),
                            str(client_index) + ' test_accTop5': accTop5_avg.value(),
                            }
            # wandb.log({str(client_index)+" Test/Loss": test_metrics[str(client_index)+' test_loss']})
            # wandb.log({str(client_index)+" Test/AccTop1": test_metrics[str(client_index)+' test_accTop1']})
            acc_all.append(accTop1_avg.value())
        wandb.log({"local mean Test/AccTop1": float(np.mean(np.array(acc_all)))})

        wandb.finish()
