from functools import reduce
from math import exp


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


class Node(object):
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        layer_index：节点所属的层的编号
        node_index：节点的编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        """
        设置节点的输出值，如果节点属于输入层会用到此函数
        """
        self.output = output

    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加一个到上游节点的连接
        """
        self.upstream.append(conn)

    def calc_output(self):
        """
        计算节点输出
        """
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)
