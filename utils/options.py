#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
 
    # training arguments
    parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")#训练的轮数
    parser.add_argument('--bs', type=int, default=128, help="test batch size")#测试时的批大小
    parser.add_argument('--lr', type=float, default=0.25, help="learning rate")#初始学习率
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay size")#学习率的衰减率
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')#特定的epoch下降学习率
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")#随机梯度下降（SGD）的动量
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="sgd weight decay")#权重衰减
    parser.add_argument('--feature_dim', type=int, help = 'feature dimension', default=128)#特征维度
    
    # FL arguments
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")#指定用户的数量，通常用于分布式学习中模拟不同的客户端
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")#指定参与训练的客户端所占总用户的比例
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")#指定每个客户端（本地）训练的轮数，本地迭代次数
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")#每个客户端（本地）使用的批大小
    
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--no-iid', dest='iid', action='store_false', help='Data is not iid')
    parser.add_argument('--num_shards', type=int, default=200, help="number of shards")#数据集的分片数量
    
    # noise arguments
    parser.add_argument('--noise_type',type=str, default='symmetric', choices=['symmetric', 'pairflip', 'clean'], help='noise type of each clients')
    parser.add_argument('--noise_rate', type=float, default=0.2,  help="noise rate of each clients")
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')#参数用于控制分布式学习中的一个阶段性调整的参数 T_k
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    
    # "Robust Federated Learning with Noisy Labels" arguments
    parser.add_argument('--T_pl', type=int, help = 'T_pl: When to start using global guided pseudo labeling', default=30)#这个参数用于设置何时开始使用全局引导伪标签
    parser.add_argument('--lambda_cen', type=float, help = 'lambda_cen', default=1.0)#中心损失的权重
    parser.add_argument('--lambda_e', type=float, help = 'lambda_e', default=0.8)#设置边界框损失的权重
    
    # Experiment arguments
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")#指定保存实验结果的目录名称

    args = parser.parse_args()
    return args