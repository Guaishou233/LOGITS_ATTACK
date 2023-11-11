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
    parser.add_argument("--digest_epoch", type=int, default=1)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--public_epoch", type=int, default=1)
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



    def aggregate(self, scores_cache: List[torch.Tensor]) -> List[torch.Tensor]:
        consensus = []
        for scores in zip(*scores_cache):
            consensus.append(torch.stack(scores, dim=-1).mean(dim=-1))
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
        # torch.autograd.set_detect_anomaly(True)

        # 每个客户训练公共数据
        for client_index, model in enumerate(self.client_models):
            # compute on public
            # 尝试加载模型参数
            model_params_file = path + f'{client_index}.pth'
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
                    print(log_probs.dtype)
                    loss = self.criterion_CE(log_probs, labels)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            client_params = model.state_dict()
            torch.save(client_params, f"{client_index}.pth")
        #测试公共数据精度
        public_acc_all = []
        for client_index, model in enumerate(self.client_models):
            #监控数据
            public_loss_avg = utils.RunningAverage()
            public_accTop1_avg = utils.RunningAverage()
            public_accTop5_avg = utils.RunningAverage()

            for batch_idx, (images, labels) in enumerate(public_test_data[0]):
                images, labels = images.to(self.device), torch.tensor(labels, dtype=torch.long).to(self.device)
                log_probs = model(images)
                loss = self.criterion_CE(log_probs, labels)
                # Update average loss and accuracy
                metrics = utils.accuracy(log_probs, labels, topk=(1, 5))
                #选择一个破坏者
                if client_index == 0:

                    # Get the top 2 values and indices for each sample
                    values, indices = torch.topk(log_probs, k=2, dim=1)

                    # Sort the indices
                    sorted_indices = torch.argsort(indices, dim=1)

                    # Gather the values and indices
                    gathered_values = torch.gather(values, dim=1, index=sorted_indices)
                    gathered_indices = torch.gather(indices, dim=1, index=sorted_indices)

                    # Swap the values and indices
                    swapped_values = torch.cat([gathered_values[:, 1:], gathered_values[:, :1]], dim=1)
                    swapped_indices = torch.cat([gathered_indices[:, 1:], gathered_indices[:, :1]], dim=1)

                    # Scatter the values and indices
                    scattered_values = torch.zeros_like(log_probs)
                    scattered_indices = torch.zeros_like(log_probs)


                    scattered_values.scatter_(dim=1, index=swapped_indices, src=swapped_values)
                    scattered_indices.scatter_(dim=1, index=swapped_indices, src=swapped_indices)

                    # Repeat the process for all dimensions
                    for i in range(2, 10):
                        values, indices = torch.topk(log_probs, k=2, dim=i)
                        sorted_indices = torch.argsort(indices, dim=i)
                        gathered_values = torch.gather(values, dim=i, index=sorted_indices)
                        gathered_indices = torch.gather(indices, dim=i, index=sorted_indices)
                        swapped_values = torch.cat([gathered_values[:, 1:], gathered_values[:, :1]], dim=i)
                        swapped_indices = torch.cat([gathered_indices[:, 1:], gathered_indices[:, :1]], dim=i)
                        scattered_values.scatter_(dim=i, index=swapped_indices, src=swapped_values)
                        scattered_indices.scatter_(dim=i, index=swapped_indices, src=swapped_indices)

                    # The final output tensor
                    log_probs = scattered_values

                scores_cache.append(log_probs)

                # only one element tensors can be converted to Python scalars
                public_accTop1_avg.update(metrics[0].item())
                public_accTop5_avg.update(metrics[1].item())
                public_loss_avg.update(loss.item())
                wandb.log({f"public top1 test Model {client_index} Accuracy": public_accTop1_avg.value()})
                # wandb.log({f"public loss test Model {client_index} Accuracy": public_loss_avg.value()})
            public_acc_all.append(public_accTop1_avg.value())
        wandb.log({"public mean Test/AccTop1": float(np.mean(np.array(public_acc_all)))})

        # aggregate
        self.consensus = self.aggregate(scores_cache)

        # digest & revisit
        for client_index, model in enumerate(self.client_models):
            model_params_file = path + f'{client_index}.pth'
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
                    loss = self.mse_criterion(log_probs, self.consensus[client_index])
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
