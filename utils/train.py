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


def fedAvg(w):
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
        #创建一个全零的张量用作伪标签，长度与数据集大小相同，并根据参数中指定的设备存储
        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.args.device)
        self.sim = torch.nn.CosineSimilarity(dim=1) 
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')#交叉熵损失函数的对象，用于计算损失
        self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#数据加载器，用于加载本地数据集的子集
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=1, shuffle=True)
    #计算联邦学习中的损失函数   
    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')#创建一个均方误差（MSE）损失函数的对象
        ce = torch.nn.CrossEntropyLoss()#交叉熵损失函数
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)
        
        L_c = ce(logit[small_loss_idxs], new_labels)#交叉熵损失 151
        L_cen = torch.sum(mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))#类特征损失函数L cen k
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))#预测结果的熵正则化
        
        lambda_e = self.args.lambda_e#该参数用于指定熵正则化损失 L_e 在总损失中的权重
        lambda_cen = self.args.lambda_cen#该参数用于指定中心特征损失 L_cen 在总损失中的权重
        if self.args.g_epoch < self.args.T_pl:#动态调整
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl
        
        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)
    #用于根据预测结果和真实标签来获取损失较小的样本索引列表   
    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)##交叉熵损失函数的对象
        ind_sorted = np.argsort(loss.data.cpu())
        loss_sorted = loss[ind_sorted]
        #根据指定的遗忘率，通过切片操作 ind_sorted[:num_remember] 获取损失较小的样本的索引列表
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update=ind_sorted[:num_remember]
        
        return ind_update#D^k
        
    def train(self, net, f_G, client_num):
        #创建一个随机梯度下降（SGD）优化器，用于更新模型的参数
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        epoch_loss = []
        
        net.eval()#将模型设置为评估模式
        f_k = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.args.device)
        # 获得全局引导伪标签global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        
        with torch.no_grad():#上下文管理器，表示在该范围内的操作不会被追踪梯度
            #对训练数据集进行批处理的循环
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)#神经网络 net 对图像进行前向传播，得到预测结果 logit 和特征向量 feature
                self.pseudo_labels[idxs] = torch.argmax(logit)#根据模型的预测结果 logit，生成伪标签   
                if self.args.g_epoch == 0:
                    f_k[labels] += feature#将每个样本的特征向量 feature 累加到对应类别的特征向量
                    n_labels[labels] += 1
            
        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1           
            f_k = torch.div(f_k, n_labels)#对类中心特征进行归一化处理
        else:
            f_k = f_G#全局类中心 f_G 赋值给 f_k，这样在后续的训练中就会使用全局特征来更新模型

        net.train()#训练模式
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):#对训练数据集 self.ldr_train 进行批处理迭代
                net.zero_grad()        
                images, labels, idx = batch
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logit, feature = net(images)#图像数据 images 和标签 labels 送入模型进行前向计算，得到预测结果 logit 和特征 feature
                feature = feature.detach()#feature从计算图中分离
                f_k = f_k.to(self.args.device)
                #调用方法，从当前批次中选择损失较小的样本索引
                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate)
                #生成伪标签并进行掩码标记
                y_k_tilde = torch.zeros(self.args.local_bs, device=self.args.device)
                mask = torch.zeros(self.args.local_bs, device=self.args.device)
                for i in small_loss_idxs:
                    #通过计算特征向量与所有类中心的相似度，选取相似度最高的类作为伪标签
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, self.args.feature_dim))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1#伪标签与真实标签相同，则在相应的mask标记为 1，表示该样本可信。
 
                # 使用伪标签pseudo-labels时
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:    
                        self.pseudo_labels[idx[i]] = labels[i]#给self.pseudo_labels 赋值
                
                #根据掩码mask来选择是否使用真实标签还是伪标签Lc k
                new_labels = mask[small_loss_idxs]*labels[small_loss_idxs] + (1-mask[small_loss_idxs])*self.pseudo_labels[idx[small_loss_idxs]]
                new_labels = new_labels.type(torch.LongTensor).to(self.args.device)
                
                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                
                loss.backward()#用来计算损失函数关于模型参数的梯度
                optimizer.step()#更新模型参数

                #全局类中心f_kj_hat 
                f_kj_hat = torch.zeros(self.args.num_classes, self.args.feature_dim, device=self.args.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.args.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                #更新本地类中心f_k
                one = torch.ones(self.args.num_classes, 1, device=self.args.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())#当前迭代的损失（float型）添加到列表 batch_loss 中
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))#这一行计算了当前epoch的平均损失（将所有batch的损失值相加除以batch的数量）
        #返回当前模型的参数字典，所有epoch的平均损失，更新后的全局类中心
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k         
