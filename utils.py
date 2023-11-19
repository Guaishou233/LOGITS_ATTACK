"""
Reference:
https://github.com/FedML-AI/FedML
"""

import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total = self.total+val
        self.steps = self.steps+1

    def value(self):
        return self.total / float(self.steps)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)
        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)
        return loss


def change_logits(log_probs):
    num_columns = log_probs.size(1)
    # 使用 torch.topk 获取每一行的值和索引
    values, indices = torch.topk(log_probs, k=num_columns, dim=1)
    # print("-----values-------indices-------")
    # print(values)
    # print(indices)
    # changed_indices = None
    # for i in range(0, num_columns, 2):
    # # 交换索引的位置
    # swapped_indices = torch.cat([indices[:, i + 1:i + 2], indices[:, i:i + 1]], dim=1)
    # # print("-----swapped_indices--------------")
    # # print(swapped_indices)
    # if i == 0:
    #     changed_indices = swapped_indices
    # else:
    #     changed_indices = torch.cat([changed_indices, swapped_indices], dim=1)
    # # print("-----changed_indices--------------")
    # # print(changed_indices)
    # # 更新 log_probs 中的索引
    changed_indices = torch.cat((indices[:, 1:], indices[:, 0].unsqueeze(1)), dim=1)

    values = values.to(log_probs)
    log_probs = log_probs.scatter(dim=1, index=changed_indices, src=values)
    # print("-----changed_log_probs--------------")
    # print(log_probs)
    return log_probs

def repalceLogitsWith0(log_probs):
    # 将所有 logits 设置为0
    log_probs = log_probs.zero_()


    return log_probs

def replace_logits_with_random(log_probs):
    # 指定均值和方差
    mean = 0.0
    std = 1.0

    # 生成具有指定均值和方差的随机数
    random_logits = torch.normal(mean=mean, std=std, size=log_probs.size())

    # 将输入的 logits 替换为随机数
    log_probs.data.copy_(random_logits)


    return log_probs
