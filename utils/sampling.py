#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np

def sample_iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))#函数从所有数据索引中随机选择num_items个索引（不重复）
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def sample_noniid(labels, num_users, num_shards, num_imgs):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param num_users:
    :return:
    """
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)#包含所有图像索引的数组

    # sort labels
    idxs_labels = np.vstack((idxs, labels))#将图像索引和对应的标签堆叠在一起
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#将图像的索引和对应的标签按照标签进行排序
    idxs = idxs_labels[0, :]

    # divide and assign为每个用户随机选择两个不同的分片索引，然后将这两个分片中的所有图像索引加入到对应用户的数据索引数组中
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    # data type cast将每个用户的数据索引数组转换为整数类型，并将其转换为Python列表 
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype('int').tolist()

    return dict_users
