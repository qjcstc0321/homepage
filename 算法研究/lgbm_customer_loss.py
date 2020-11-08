# coding: utf-8

"""
LightGBM自定义损失函数
"""


import numpy as np


def logloss(preds, train_data):
    """
    二分类交叉熵损失函数
    L(y, p) = -(ylogp + (1-y)log(1-p))
    """
    y = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    # loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    loss = np.where(y == 1, -np.log(p), -np.log(1 - p))

    return 'logloss', np.mean(loss), False


def logloss_fobj(preds, train_data):
    """
    二分类交叉熵损失函数的一阶导和二阶导
    L(y, p) = -(ylogp + (1-y)log(1-p))
    L' = p - y
    L'' = p(1 - p)
    """
    y = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    grad = p - y
    hess = p * (1.0 - p)

    return grad, hess


def smooth_logloss(preds, train_data):
    """
    带标签平滑的二分类交叉熵损失函数,
    当y=1时, y'=1-ε, 当y=0时, y'=ε
    L(y, p) = -(y'logp + (1-y')log(1-p))
    """
    eps = 0.02    # 默认值, 可以根据实际情况修改
    y = np.abs(train_data.get_label() - eps)
    p = 1.0 / (1.0 + np.exp(-preds))
    # loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    loss = np.where(y == 1, -np.log(p), -np.log(1 - p))

    return 'smooth_logloss', np.mean(loss), False


def labelsmooth_fobj(preds, train_data):
    """
    带标签平滑的二分类交叉熵损失函数的一阶导和二阶导
    当y=1时, y'=1-ε, 当y=0时, y'=ε
    L(y, p) = -(y'logp + (1-y')log(1-p))
    L' = p - y'
    L'' = p(1 - p)
    """
    eps = 0.018
    y = np.abs(train_data.get_label() - eps)
    p = 1.0 / (1.0 + np.exp(-preds))
    grad = p - y
    hess = p * (1.0 - p)

    return grad, hess


def focalloss(preds, train_data):
    """
    focal loss
    """
    a = 0.5
    r = 1
    y = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    loss = np.where(y == 1, -a * (1-p)**r * np.log(p), -(1-a) * p**r * np.log(1-p))

    return 'focalloss', np.mean(loss), False


def focalloss_fobj(preds, train_data):
    """
    focal loss的一阶导数和二阶导数
    """
    a = 0.5
    r = 1
    y = train_data.get_label()
    p1 = 1.0 / (1.0 + np.exp(-preds))
    p0 = 1 - p1
    logp1 = np.log(p1)
    logp0 = np.log(p0)
    grad = np.where(y == 1,
                    -a * p0**r * (p0 - r*p1*logp1),
                    -(1-a) * p1**r * (r*p0*logp0 - p1))
    hess = np.where(y == 1,
                    -a * p1 * p0**r * ((r*r*p1 - r*p0)*logp1 - (2*r+1)*p0),
                    -(1-a) * p1**r * p0 * ((r*r*p0 - r*p1)*logp0 - (2*r+1)*p1))

    return grad, hess
