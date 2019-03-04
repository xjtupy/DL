#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/2 16:58
# @Author: Vincent
# @File  : sigmoid.py

import numpy as np


class SigmoidActivator(object):
    """
    Sigmoid激活函数类
    """

    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)
