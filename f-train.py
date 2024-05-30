#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy

#创建并返回一个包含了各个用户本地更新对象的列表，每个本地更新对象用于执行一个用户的本地模型更新操作
def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(LocalUpdateRFL(**local_update_args))

    return local_update_objects


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
            
    return w_avg

#用于创建一个分割后的数据集对象，该数据集对象包含了原始数据集中特定索引范围内的样本
class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):#用于按索引获取数据集中的单个样本
        item = int(item)
        image, label = self.dataset[self.idxs[item]]#使用 idxs 中的索引从原始数据集中获取对应的图像和标签

        return image, label, self.idxs[item]
        

class LocalUpdateRFL:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)


    #用于根据预测结果和真实标签来获取损失较小的样本索引列表   
    def train(self, net):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []

        net.train()
        for _ in range(self.args.local_ep):
            batch_loss = []
            for _ , batch in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels, _ = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, _ = net(images)
                loss = self.loss_func(logit, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
