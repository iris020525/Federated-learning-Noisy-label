#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import numpy as np
import random

import torchvision
from torchvision import transforms
import torch

from data.cifar import CIFAR10
from data.mnist import MNIST
from model.Nets import CNN
from utils.logger import Logger
from utils.sampling import sample_iid, sample_noniid
from utils.options import args_parser
from utils.noisify import noisify_label
from utils.train import get_local_update_objects, FedAvg
from utils.test import test_img

import time
import torch
import torch.nn as nn
import numpy as np
import copy
import time

if __name__ == '__main__':

    start = time.time()
    #解析参数
    args = args_parser()
    args.device = torch.device('cpu')#指定使用CPU运行
        #'cuda:{}'.format(args.gpu)
        #if torch.cuda.is_available() and args.gpu != -1
        #else 'cpu',
    
#获取args对象的属性和值，然后通过items转换为一个键-值对的元组，并将它们存储在变量 x 中
    for x in vars(args).items():
        print(x)

    #if not torch.cuda.is_available():
    #    exit('ERROR: Cuda is not available!')
    #print('torch version: ', torch.__version__)
    #print('torchvision version: ', torchvision.__version__)

    # Seed随机种子
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    ##############################
    #加载数据集，拆分用户
    ##############################
    
    if args.dataset == 'mnist':
        import urllib #from six.moves import urllib会报错未解析six

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        #数据预处理：训练集进行了随机裁剪和水平翻转
        trans_mnist_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        #数据预处理：对验证集和测试集进行了标准的转换（Tensor 格式）和归一化操作。
        trans_mnist_val = transforms.Compose([
            transforms.ToTensor(),
            #将每个像素的值减去均值（0.1307），除以标准差（0.3081），使得输入数据的分布更加接近标准正态分布
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset_train = MNIST(
            root='D:\Robust-Federated-Learning-with-Noisy-Labels-main\data\mnist',
            download=True, 
            train=True,
            transform=trans_mnist_train,
        )
        dataset_test = MNIST(
            root='D:\Robust-Federated-Learning-with-Noisy-Labels-main\data\mnist',
            download=True,
            train=False,
            transform=trans_mnist_val,
        )
        num_classes = 10
        input_channel = 1
    
    elif args.dataset == 'cifar':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10(
            root='./data/cifar',
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=True,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10
        input_channel = 3

    else:
        raise NotImplementedError('Error: unrecognized dataset')
    #提取标签和设置相关参数（数据分片图像数量，图像大小，类别数目）
    labels = np.array(dataset_train.train_labels)
    num_imgs = len(dataset_train) // args.num_shards
    args.img_size = dataset_train[0][0].shape  # used to get model
    args.num_classes = num_classes

    # Sample users (iid / non-iid)
    if args.iid:
        dict_users = sample_iid(dataset_train, args.num_users)
    else:
        dict_users = sample_noniid(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
            num_imgs=num_imgs,
        )

    ##############################
    # 在数据中添加噪声标签
    ##############################
    #dict_users[i]:存储客户端client i 拥有的数据data索引index
    if args.noise_type != "clean":
        for user in range(args.num_users):
            data_indices = list(copy.deepcopy(dict_users[user]))

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)#用随机函数将数据索引列表随机打乱

            noise_index = int(len(data_indices) * args.noise_rate)#计算要添加噪声的数据索引数量（计算数据索引长度乘以噪声率）

            for d_idx in data_indices[:noise_index]:#遍历需要添加噪声的索引列表
                true_label = dataset_train.train_labels[d_idx]#获取数据集dataset_train中该索引对应的真实标签d_idx
                noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=args.noise_type)#函数根据噪声类型、类别数量等参数生成一个带有噪声的标签
                dataset_train.train_labels[d_idx] = noisy_label#将真实标签d_idx替换为噪声标签

    # 创建训练数据加载器和测试数据加载器，用于后续训练和测试的迭代中
    log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)

    ##############################
    # 建立模型
    ##############################
    net_glob = CNN(input_channel=input_channel)#创建了一个名为 net_glob 的卷积神经网络（CNN）模型实例
    net_glob = net_glob.to(args.device)
    print(net_glob)

    ##############################
    # 训练
    ##############################
    logger = Logger(args)

    forget_rate_schedule = []  # 空列表用于存储遗忘率（forget rate）的变化规律
                
    forget_rate = args.forget_rate
    exponent = 1
    forget_rate_schedule = np.ones(args.epochs) * forget_rate  # 每个元素都被初始化为 forget_rate 的值
    # 根据指定的参数 args.num_gradual，通过线性插值的方式将前 args.num_gradual 个遗忘率的值从 0 逐渐变化到 forget_rate 的指数次幂
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)

    # 初始化全局类中心特征f_G
    # f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)

    # 初始化本地更新对象
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        net_glob=net_glob,
    )
   ##############################################################
start = time.time()
for epoch in range(args.epochs):
    local_losses = [] # 用于存储本地损失值
    local_weights = [] # 用于存储本地模型权重
    
    args.g_epoch = epoch # 记录当前是第几个epoch
    
    # 学习率衰减
    if (epoch + 1) in args.schedule:
        print("Learning Rate Decay Epoch {}".format(epoch + 1))
        print("{} => {}".format(args.lr, args.lr * args.lr_decay)) # 打印学习率衰减前后的学习率值
        args.lr *= args.lr_decay # 学习率乘以衰减因子
    
    if len(forget_rate_schedule) > 0: # 检查是否存在遗忘率（forget rate）的计划
        args.forget_rate = forget_rate_schedule[epoch] # 获取相应的遗忘率并将其赋值给前者
    
    m = max(int(args.frac * args.num_users), 1) # 计算参与训练的用户数量
    idxs_users = np.random.choice(range(args.num_users), m, replace=False) # 随机选择m个用户索引。参数 replace=False 确保所选用户不重复
    
    # 本地更新
    for idx in idxs_users:
        local = local_update_objects[idx] # 获取对应的本地更新对象并将其赋值给变量 local
        local.args = args
        # 调用本地更新对象 local 的 train 方法进行本地模型训练
        w, loss = local.train(copy.deepcopy(net_glob).to(args.device))
        local_weights.append(copy.deepcopy(w)) # 参数w添加到local_weights列表中
        local_losses.append(copy.deepcopy(loss))
    
    # 通过FedAvg聚合，更新全局权重
    w_glob = FedAvg(local_weights)
    net_glob.load_state_dict(w_glob)
    ''''
    sim = torch.nn.CosineSimilarity(dim=1) # 计算余弦相似度
    tmp = 0
    w_sum = 0
    for i in local_weights:
        sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
        w_sum += sim_weight
        tmp += sim_weight * i
    f_G = torch.div(tmp, w_sum) # 通过将加权后的本地类中心特征向量之和tmp除以相似度权重的加权和w_sum，来更新全局特征
    '''
    # 计算训练集和测试集上的准确率和损失
    train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
    test_acc, test_loss = test_img(net_glob, log_test_data_loader, args)
    results = dict(train_acc=train_acc, train_loss=train_loss,
                   test_acc=test_acc, test_loss=test_loss,)
    
    print('Round {:3d}'.format(epoch)) # 打印当前轮次（epoch）的序号
    print(' - '.join([f'{k}: {v:.6f}' for k, v in results.items()])) # 打印每轮训练/测试的结果
    
    logger.write(epoch=epoch + 1, **results) # 当前轮次的结果写入日志文件

logger.close()

print("time :", time.time() - start) # 打印总执行时间