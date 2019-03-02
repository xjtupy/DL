#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/3/2 15:58
# @Author: Vincent
# @File  : connections.py


class Connections(object):
    """
    Connections对象，提供Connection集合操作。
    """

    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)
