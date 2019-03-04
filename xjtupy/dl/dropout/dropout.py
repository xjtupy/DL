#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/3 15:02
# @Author: Vincent
# @File  : dropout.py


# coding:utf-8
import numpy as np


def dropout(x, level):
    """
    dropout函数的实现
    :param x: 当前输入值
    :param level: 隐藏神经元的概率
    :return:
    """
    if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
        raise ValueError('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level

    # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape)
    print(random_tensor)

    x *= random_tensor
    print(x)
    x /= retain_prob

    return x


# 对dropout的测试，了解一个输入x向量，经过dropout的结果
x = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
print(dropout(x, 0.4))
