#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/2 15:55
# @Author: Vincent
# @File  : layer.py
from DL.xjtupy.dl.network.constnode import ConstNode
from DL.xjtupy.dl.network.node import Node


class Layer(object):
    """
    层类
    """
    def __init__(self, layer_index, node_count):
        """
        初始化一层
        layer_index: 层编号
        node_count: 层所包含的节点个数
        """
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        """
        设置层的输出。当层是输入层时会用到。
        """
        data = list(data)
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        """
        计算层的输出向量
        """
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        """
        打印层的信息
        """
        for node in self.nodes:
            print(node)