#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#评估模型在测试集上的性能（accuracy, test_loss）
def test_img(net_g, data_loader, args):
    net_g.eval()
    test_loss = 0
    correct = 0#正确预测数
    n_total = len(data_loader.dataset)
    
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs, _ = net_g(data)#数据 data 提交给模型 net_g 进行预测
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()#用模型的预测结果 log_probs 和真实标签 target计算批次的交叉熵损失，并将其添加到总的测试损失中。
        y_pred = log_probs.data.max(1, keepdim=True)[1] #获取预测结果中概率最大的类别的索引，即找到每个样本预测概率最高的类别。
        correct += y_pred.eq(target.data.view_as(y_pred)).float().sum().item()#计算批次中预测正确的样本数

    test_loss /= n_total
    accuracy = 100.0 * correct / n_total

    return accuracy, test_loss

