# coding: utf-8

"""
define tree structure
"""


import numpy as np


class Node(object):
    def __init__(self, depth, samples, impurity):
        self.depth = depth
        self.samples = samples
        self.impurity = impurity
        self.value = None
        self.split_var = None
        self.split_val = None
        self.gain = None
        self.left = None
        self.right = None

    def set_as_terminal(self, value):
        """
        将节点设置为叶结点
        """
        self.value = value

    def split(self, split_var, split_val, split_gain, left, right):
        """
        节点分裂
        """
        self.split_var = split_var
        self.split_val = split_val
        self.gain = split_gain
        self.left = left
        self.right = right


class Tree(object):
    def __init__(self):
        self.node_count = 0
        self.leaf_nodes = 0
        self.root = None

    @property
    def max_depth(self):
        return max(self._traverse('depth'))

    @property
    def node_samples(self):
        return self._traverse('samples')

    @property
    def node_impurity(self):
        return self._traverse('impurity')

    @property
    def split_variables(self):
        return self._traverse('split_var')

    @property
    def split_values(self):
        return self._traverse('split_val')

    @property
    def terminal_values(self):
        return self._traverse('value')

    def _traverse(self, attr):
        """
        DFS遍历树的节点, 获取指定的节点属性
        """
        if self.root is None:
            raise TypeError('tree is not be bulided')
        stack = [self.root]
        result = []
        while stack:
            cur_node = stack.pop()
            if getattr(cur_node, attr) is None:
                result.append(-1)
            else:
                result.append(getattr(cur_node, attr))
            if cur_node.right is not None:
                stack.append(cur_node.right)
            if cur_node.left is not None:
                stack.append(cur_node.left)

        return result

    def add_root(self, samples, impurity):
        """
        添加根节点
        Parameters
        ----------
        samples: int
            样本数量
        impurity: float
            样本信息不纯度
        """
        self.root = Node(depth=0, samples=samples, impurity=impurity)
        self.node_count += 1

    def get_terminal_value(self, x):
        """
        获取样本归属的叶节点输出
        Parameters
        ----------
        x: ndarray

        Return
        ------
        float
        """
        if self.root is None:
            raise TypeError('tree is not be bulided')
        cur_node = self.root
        while True:
            if cur_node.value is not None:
                return cur_node.value

            if x[cur_node.split_var] <= cur_node.split_val:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
