# coding: utf-8

"""
Classifier and Regression Tree algorithm
"""


import numpy as np
from treebase import Node, Tree


class ClassifierTree(object):
    """
    决策树分类模型
    Parameters
    ----------
    criterion: str, options['gini', 'entropy']
        度量样本不纯度的准则
    max_depth: int, default -1
        树的最大深度, -1表示不限制最大深度
    max_leaf_nodes: int, default -1
        叶节点的最大个数
    min_samples_split: int, default 2
        节点分裂前的最小样本数
    min_impurity_split: float, default 1e-6
        节点分裂前的最小信息不纯度
    min_split_gain: float, default 1e-6
        节点分裂的最小增益
    min_samples_in_leaf: int, default 1
        叶节点的最小样本数

    Attributes
    ----------
    tree_: Tree object
        树结构
    feature_importance_: ndarray
        每个特征的分裂增益
    """
    def __init__(self, criterion, max_depth=-1, max_leaf_nodes=-1, min_samples_split=2, min_impurity_split=1e-6, min_split_gain=1e-6,
                 min_samples_in_leaf=1):

        if criterion not in ['gini', 'entropy']:
            raise ValueError('"criterion" must be "gini" or "entropy"')
        if max_depth == 0:
            raise ValueError('"max_depth" must be greater than 0')
        if (max_leaf_nodes == 0) or (max_leaf_nodes == 1):
            raise ValueError('"max_depth" must be greater than 1')
        if min_samples_split <= 1:
            raise ValueError('"min_samples_split" must be greater than 1')
        if min_impurity_split <= 0:
            raise ValueError('"min_impurity_split" must be greater than 0')
        if min_split_gain <= 0:
            raise ValueError('"min_split_gain" must be greater than 0')
        if min_samples_in_leaf <= 0:
            raise ValueError('"min_split_gain" must be greater than 0')

        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.min_split_gain = min_split_gain
        self.min_samples_in_leaf = min_samples_in_leaf

    def _calc_impurity(self, y):
        """
        计算样本信息不纯度
        """
        p1 = 1.0 * np.sum(y) / len(y)
        p0 = 1.0 - p1
        if self.criterion == 'gini':
            return 1.0 - p0*p0 - p1*p1

        elif self.criterion == 'entropy':
            if p0 == 0.0 or p1 == 0.0:
                return 0.0
            else:
                return -p1*np.log2(p1) - p0*np.log2(p0)

    def _calc_gain(self, node, sample_slice, condition, y):
        """
        计算节点的分裂增益
        """
        slice1 = sample_slice[condition]
        slice2 = sample_slice[~condition]
        prop1 = 1.0 * slice1.size / sample_slice.size
        prop2 = 1.0 * slice2.size / sample_slice.size
        gain = node.impurity - prop1 * self._calc_impurity(y[slice1]) - prop2 * self._calc_impurity(y[slice2])

        return gain, slice1, slice2

    def _check_node_cond(self, node):
        """
        检查当前节点是否满足以下条件:
        1. 节点的样本数是否满足min_samples_split
        2. 节点的深度是否满足max_depth
        3. 节点的样本不纯度是否满足min_impurity_split
        """
        if node.samples < self.min_samples_split:
            return False
        if (self.max_depth > 0) and (node.depth >= self.max_depth):
            return False
        if node.impurity < self.min_impurity_split:
            return False

        return True

    def _check_split_cond(self, split_gain, slice_l, slice_r):
        """
        检查当前节点分裂后是否满足以下条件:
        1. 分裂增益是否大于min_split_gain
        2. 分裂后的左右子节点样本数是否满足min_samples_in_leaf
        """
        if split_gain < self.min_split_gain:
            return False
        if (len(slice_l) < self.min_samples_in_leaf) or (len(slice_r) < self.min_samples_in_leaf):
            return False

        return True

    def train(self, data, label):
        """
        Parameters
        ----------
        data: ndarray
            样本特征矩阵
        label: ndarray
            样本标签

        Return
        ------
        self: ClassifierTree object
        """
        if data.shape[0] != label.shape[0]:
            raise ValueError('Number of samples does not match number of labels')
        if np.unique(label).size > 2:
            raise ValueError('label must be two class')

        self.tree_ = Tree()
        self.feature_importance_ = np.zeros(data.shape[1])
        self.tree_.add_root(samples=label.shape[0], impurity=self._calc_impurity(label))
        stack = [(self.tree_.root, np.arange(0, data.shape[0]))]    # 将节点和样本切片绑定在一起入栈
        while stack:
            cur_node, node_slice = stack.pop()
            if self._check_node_cond(cur_node) and ((self.max_leaf_nodes == -1) or (self.tree_.leaf_nodes + 2 <= self.max_leaf_nodes)):
                max_gain = 0
                slice_l = None
                slice_r = None
                split_var = None
                split_val = None
                for i in np.arange(0, data.shape[1]):
                    x_i = X[node_slice, i]
                    unique_values = np.unique(x_i)
                    if len(unique_values) == 1:
                        continue
                    for t in unique_values[:-1]:
                        gain, slice1, slice2 = self._calc_gain(node=cur_node, sample_slice=node_slice, condition=(x_i <= t), y=label)
                        if gain > max_gain:
                            max_gain = gain
                            slice_l, slice_r = slice1, slice2
                            split_var, split_val = i, t

                if self._check_split_cond(max_gain, slice_l, slice_r):
                    left_child = Node(depth=cur_node.depth+1, samples=len(slice_l), impurity=self._calc_impurity(label[slice_l]))
                    right_child = Node(depth=cur_node.depth+1, samples=len(slice_r), impurity=self._calc_impurity(label[slice_r]))
                    cur_node.split(split_var=split_var, split_val=split_val, split_gain=max_gain, left=left_child, right=right_child)
                    stack.append((cur_node.right, slice_r))
                    stack.append((cur_node.left, slice_l))
                    self.tree_.node_count += 2
                    self.feature_importance_[split_var] += max_gain
                else:
                    cur_node.set_as_terminal(value=label[node_slice].mean())
                    self.tree_.leaf_nodes += 1
            else:
                cur_node.set_as_terminal(value=label[node_slice].mean())
                self.tree_.leaf_nodes += 1

        return self

    def predict(self, data):
        """
        Parameters
        ----------
        data: ndarray
            样本特征矩阵

        Return
        ------
        p: ndarray
            样本属于1的概率
        """
        if 'tree_' not in clf.__dict__:
            raise TypeError('model is not fitted yet')
        p = np.apply_along_axis(self.tree_.get_terminal_value, 1, data)

        return p


class RegressorTree(object):
    """
    决策树回归模型
    Parameters
    ----------
    max_depth: int, default -1
        树的最大深度, -1表示不限制最大深度
    max_leaf_nodes: int, default -1
        叶节点的最大个数
    min_samples_split: int, default 2
        节点分裂前的最小样本数
    min_impurity_split: float, default 1e-6
        节点分裂前的最小信息不纯度
    min_split_gain: float, default 1e-6
        节点分裂的最小增益
    min_samples_in_leaf: int, default 1
        叶节点的最小样本数

    Attributes
    ----------
    tree_: Tree object
        树结构
    feature_importance_: ndarray
        每个特征的分裂增益
    """
    def __init__(self, max_depth=-1, max_leaf_nodes=-1, min_samples_split=2, min_impurity_split=1e-6, min_split_gain=1e-6,
                 min_samples_in_leaf=1):

        if max_depth == 0:
            raise ValueError('"max_depth" must be greater than 0')
        if (max_leaf_nodes == 0) or (max_leaf_nodes == 1):
            raise ValueError('"max_depth" must be greater than 1')
        if min_samples_split <= 1:
            raise ValueError('"min_samples_split" must be greater than 1')
        if min_impurity_split <= 0:
            raise ValueError('"min_impurity_split" must be greater than 0')
        if min_split_gain <= 0:
            raise ValueError('"min_split_gain" must be greater than 0')
        if min_samples_in_leaf <= 0:
            raise ValueError('"min_split_gain" must be greater than 0')

        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.min_split_gain = min_split_gain
        self.min_samples_in_leaf = min_samples_in_leaf

    def _calc_variance(self, y):
        """
        计算样本的方差
        """
        mean = np.mean(y)

        return np.sum((y - mean)**2) / len(y)

    def _calc_gain(self, node, sample_slice, condition, y):
        """
        计算节点的分裂增益
        """
        slice1 = sample_slice[condition]
        slice2 = sample_slice[~condition]
        prop1 = 1.0 * slice1.size / sample_slice.size
        prop2 = 1.0 * slice2.size / sample_slice.size
        gain = node.impurity - prop1 * self._calc_variance(y[slice1]) - prop2 * self._calc_variance(y[slice2])

        return gain, slice1, slice2

    def _check_node_cond(self, node):
        """
        检查当前节点是否满足以下条件:
        1. 节点的样本数是否满足min_samples_split
        2. 节点的深度是否满足max_depth
        3. 节点的样本不纯度是否满足min_impurity_split
        """
        if node.samples < self.min_samples_split:
            return False
        if (self.max_depth > 0) and (node.depth >= self.max_depth):
            return False
        if node.impurity < self.min_impurity_split:
            return False

        return True

    def _check_split_cond(self, split_gain, slice_l, slice_r):
        """
        检查当前节点分裂后是否满足以下条件:
        1. 分裂增益是否大于min_split_gain
        2. 分裂后的左右子节点样本数是否满足min_samples_in_leaf
        """
        if split_gain < self.min_split_gain:
            return False
        if (len(slice_l) < self.min_samples_in_leaf) or (len(slice_r) < self.min_samples_in_leaf):
            return False

        return True

    def train(self, data, y):
        """
        Parameters
        ----------
        data: ndarray
            样本特征矩阵
        y: ndarray
            样本真实值

        Return
        ------
        self: RegressorTree object
        """
        if data.shape[0] != y.shape[0]:
            raise ValueError('Number of samples does not match number of y')

        self.tree_ = Tree()
        self.feature_importance_ = np.zeros(data.shape[1])
        self.tree_.add_root(samples=y.shape[0], impurity=self._calc_variance(y))
        stack = [(self.tree_.root, np.arange(0, data.shape[0]))]    # 将节点和样本切片绑定在一起入栈
        while stack:
            cur_node, node_slice = stack.pop()
            if self._check_node_cond(cur_node) and ((self.max_leaf_nodes == -1) or (self.tree_.leaf_nodes + 2 <= self.max_leaf_nodes)):
                max_gain = 0
                slice_l = None
                slice_r = None
                split_var = None
                split_val = None
                for i in np.arange(0, data.shape[1]):
                    x_i = X[node_slice, i]
                    unique_values = np.unique(x_i)
                    if len(unique_values) == 1:
                        continue
                    for t in unique_values[:-1]:
                        gain, slice1, slice2 = self._calc_gain(node=cur_node, sample_slice=node_slice, condition=(x_i <= t), y=y)
                        if gain > max_gain:
                            max_gain = gain
                            slice_l, slice_r = slice1, slice2
                            split_var, split_val = i, t

                if self._check_split_cond(max_gain, slice_l, slice_r):
                    left_child = Node(depth=cur_node.depth+1, samples=len(slice_l), impurity=self._calc_variance(y[slice_l]))
                    right_child = Node(depth=cur_node.depth+1, samples=len(slice_r), impurity=self._calc_variance(y[slice_r]))
                    cur_node.split(split_var=split_var, split_val=split_val, split_gain=max_gain, left=left_child, right=right_child)
                    stack.append((cur_node.right, slice_r))
                    stack.append((cur_node.left, slice_l))
                    self.tree_.node_count += 2
                    self.feature_importance_[split_var] += max_gain
                else:
                    cur_node.set_as_terminal(value=y[node_slice].mean())
                    self.tree_.leaf_nodes += 1
            else:
                cur_node.set_as_terminal(value=y[node_slice].mean())
                self.tree_.leaf_nodes += 1

        return self

    def predict(self, data):
        """
        Parameters
        ----------
        data: ndarray
            样本特征矩阵

        Return
        ------
        p: ndarray
            样本属于1的概率
        """
        if 'tree_' not in clf.__dict__:
            raise TypeError('model is not fitted yet')
        y_hat = np.apply_along_axis(self.tree_.get_terminal_value, 1, data)

        return y_hat