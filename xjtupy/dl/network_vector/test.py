#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/2 16:59
# @Author: Vincent
# @File  : test.py

import random
from functools import reduce

from numpy import *

from DL.xjtupy.dl.network_vector.network import Network


class Normalizer(object):
    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        binary = list(binary)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data = [val for val in n]
        data_set.append(data)
        labels.append(data)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    correct_ratio(net)
