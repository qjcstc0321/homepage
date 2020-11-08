# coding: utf-8

"""
梯度提升树算法，相关论文《Greedy function approximation: A gradient boosting machine》
"""


import numpy as np
from decisiontree import Node, Tree


def _sigmoid(x):
    """
    sigmoid函数
    """
    return 1 / (1 + np.exp(-x))


class LogOddsLoss(object):
    """
    对数损失函数
    """
    def __init__(self):
        self.name = 'Logloss'

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1-y_true)*np.log(1 - y_pred))

    def grad(self, y_true, y_pred):
        return np.mean(y_true - y_pred)

    def hess(self, y_true, y_pred):
        return np.mean(y_pred * (y_pred - 1))


class MeanSquaresLoss(object):
    """
    MSE损失函数
    """
    def __init__(self):
        self.name = 'MSE'

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def grad(self, y_true, y_pred):
        return np.mean(y_true - y_pred)

    def hess(self, y_true, y_pred):
        return 1


class GradBooster(object):
    def __init__(self, lossfunc, max_depth=-1, max_leaf_nodes=-1, min_samples_split=2, min_impurity_split=1e-6,
                 min_split_gain=1e-6, min_samples_in_leaf=1):
        self.loss = lossfunc
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

    def train(self, X, y, y_pred):
        """
        Parameters
        ----------
        X: numpy.ndarray
            样本特征矩阵
        y: numpy.ndarray
            样本真实值
        y_pred: numpy.ndarray
            样本上一轮的预测值

        Returns
        -------
        GradBooster object
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples does not match number of y')
        residual = y - y_pred
        self.tree_ = Tree()
        self.feature_importance_ = np.zeros(X.shape[1])
        self.tree_.add_root(samples=y.shape[0], impurity=self._calc_variance(residual))
        stack = [(self.tree_.root, np.arange(0, X.shape[0]))]    # 将节点和样本切片绑定在一起入栈
        while stack:
            cur_node, node_slice = stack.pop()
            if self._check_node_cond(cur_node) and ((self.max_leaf_nodes == -1) or (self.tree_.leaf_nodes + 2 <= self.max_leaf_nodes)):
                max_gain = 0
                slice_l = None
                slice_r = None
                split_var = None
                split_val = None
                for i in np.arange(0, X.shape[1]):
                    x_i = X[node_slice, i]
                    unique_values = np.unique(x_i)
                    if len(unique_values) == 1:
                        continue
                    for t in unique_values[:-1]:
                        gain, slice1, slice2 = self._calc_gain(node=cur_node, sample_slice=node_slice, condition=(x_i <= t), y=residual)
                        if gain > max_gain:
                            max_gain = gain
                            slice_l, slice_r = slice1, slice2
                            split_var, split_val = i, t

                if self._check_split_cond(max_gain, slice_l, slice_r):
                    left_child = Node(depth=cur_node.depth+1, samples=len(slice_l), impurity=self._calc_variance(residual[slice_l]))
                    right_child = Node(depth=cur_node.depth+1, samples=len(slice_r), impurity=self._calc_variance(residual[slice_r]))
                    cur_node.split(split_var=split_var, split_val=split_val, split_gain=max_gain, left=left_child, right=right_child)
                    stack.append((cur_node.right, slice_r))
                    stack.append((cur_node.left, slice_l))
                    self.tree_.node_count += 2
                    self.feature_importance_[split_var] += max_gain
                else:
                    terminal_value = -self.loss.grad(y[node_slice], y_pred[node_slice]) / self.loss.hess(y[node_slice], y_pred[node_slice])
                    cur_node.set_as_terminal(value=terminal_value)
                    self.tree_.leaf_nodes += 1
            else:
                terminal_value = -self.loss.grad(y[node_slice], y_pred[node_slice]) / self.loss.hess(y[node_slice], y_pred[node_slice])
                cur_node.set_as_terminal(value=terminal_value)
                self.tree_.leaf_nodes += 1

        return self

    def predict(self, data):
        """
        Parameters
        ----------
        data: numpy.ndarray
            样本特征矩阵

        Return
        ------
        p: numpy.ndarray
            样本属于1的概率
        """
        if 'tree_' not in self.__dict__:
            raise TypeError('model is not fitted yet')
        y_hat = np.apply_along_axis(self.tree_.get_terminal_value, 1, data)

        return y_hat


class GradBoostTree(object):
    def __init__(self, objective, boostround, learning_rate, max_depth=-1, max_leaf_nodes=-1, min_samples_split=2, min_samples_in_leaf=1):
        if objective not in ['binary', 'regression']:
            raise ValueError('"objective" must be "binary" or "regression"')
        if max_depth == 0:
            raise ValueError('"max_depth" must be greater than 0')
        if (max_leaf_nodes == 0) or (max_leaf_nodes == 1):
            raise ValueError('"max_depth" must be greater than 1')
        if min_samples_split <= 1:
            raise ValueError('"min_samples_split" must be greater than 1')
        if min_samples_in_leaf <= 0:
            raise ValueError('"min_split_gain" must be greater than 0')

        self.objective = objective
        self.boostround = boostround
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_in_leaf = min_samples_in_leaf
        self._booster = []
        if self.objective == 'regression':
            self.loss = MeanSquaresLoss()
        else:
            self.loss = LogOddsLoss()

    def __get_init_score(self, y):
        """
        计算F0
        """
        if self.objective == 'regression':
            return np.mean(y)
        else:
            return np.log(np.sum(y) / np.sum(1 - y))

    def train(self, X, y):
        self.init_score = self.__get_init_score(y)
        Fm = np.repeat(self.init_score, len(y))
        y_pred = _sigmoid(Fm)
        for i in range(self.boostround):
            tree = GradBooster(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,
                                     min_samples_split=self.min_samples_split, min_samples_in_leaf=self.min_samples_in_leaf,
                                     lossfunc=self.loss)
            tree = tree.train(X, y=y, y_pred=y_pred)
            Fm += self.learning_rate * tree.predict(X)    # 更新Fm
            y_pred = _sigmoid(Fm)    # 更新y_pred
            cur_loss = self.loss.loss(y, y_pred)
            print('Boost round: {0}, {1} = {2}'.format(i+1, self.loss.name, cur_loss))
            self._booster.append(tree)

        return self

    def predict(self, X):
        Fm = np.repeat(self.init_score, len(X))
        for tree in self._booster:
            Fm += self.learning_rate * tree.predict(X)

        return Fm

    def predict_proba(self, X):
        Fm = self.predict(X)

        return _sigmoid(Fm)
