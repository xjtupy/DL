#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/2 16:59
# @Author: Vincent
# @File  : network.py
from DL.xjtupy.dl.network_vector.fullConnectedLayer import FullConnectedLayer
from DL.xjtupy.dl.network_vector.sigmoid import SigmoidActivator


class Network(object):
    """
    神经网络类
    """

    def __init__(self, layers):
        """
        构造函数
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, sample):
        """
        使用神经网络实现预测
        sample: 输入样本
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        label = list(label)
        print(label)
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)
