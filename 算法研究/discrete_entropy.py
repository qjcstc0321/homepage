# coding: utf-8

"""
计算单变量的熵，多个变量的联合信息熵和互信息
"""


import pandas as pd
import numpy as np


def entropy(df, x):
    """
    计算单个离散变量的信息熵
    Parameters
    ----------
    df: pandas.DataFrame
    x: str
        变量名

    Returns
    -------
    float
    """
    if df[x].unique().size == 1:
        return 0.0
    pdf = (df[x].value_counts() / df.shape[0]).values   # 离散概率分布列

    return -(pdf * np.log2(pdf)).sum()


def joint_entropy(df, x, y, z=None):
    """
    计算2-3个离散变量的联合信息熵
    Parameters
    ----------
    df: pandas.DataFrame
    x: str
        变量名1
    y: str
        变量名2
    z: str, default None
        变量名3，此变量只允许服从0-1分布的变量

    Returns
    -------
    float
    """
    if z is not None:
        if set(df[z].unique()) != {0, 1}:
            raise ValueError('variable z be allowed values 0 and 1')
        xy_joint_freq = pd.crosstab(index=df[x], columns=df[y]).astype(float)
        count_z1 = pd.crosstab(index=df[x], columns=df[y], values=df[z], aggfunc='sum').fillna(0)
        count_z0 = xy_joint_freq - count_z1
        joint_pdf = np.array([count_z1.values, count_z0.values], dtype=float) / df.shape[0]
    else:
        joint_pdf = pd.crosstab(df[x], df[y], normalize=True).values

    joint_pdf = np.where(joint_pdf == 0, 1e-6, joint_pdf)     # laplace平滑

    return -(joint_pdf * np.log2(joint_pdf)).sum()


def conditional_entropy(df, target, given):
    """
    计算离散变量的条件熵
    H(X|Y) = H(X,Y) - H(Y)
    H(X,Y|Z) = H(X,Y,Z) - H(Z)
    H(X|Y,Z) = H(X,Y,Z) - H(Y,Z)
    Parameters
    ----------
    df: pandas.DataFrame
    target: str, list or tuple
        目标变量名
    given: str, list or tuple
        条件变量名

    Returns
    -------
    float
    """
    if isinstance(target, str):
        target = [target]
    if isinstance(given, str):
        given = [given]
    if len(target) + len(given) not in (2, 3):
        raise ValueError('the total number of variables is wrong')
    if len(target) == 0 or len(given) == 0:
        raise ValueError('no target variables or given variables')

    if len(target) + len(given) == 2:
        # 计算形如H(X|Y)的条件熵
        h_given = entropy(df, given[0])
        h_joint = joint_entropy(df, x=target[0], y=given[0])
    else:
        if len(target) == 1:
            # 计算形如H(X|Y,Z)的条件熵
            h_given = joint_entropy(df, x=given[0], y=given[1])
            h_joint = joint_entropy(df, x=target[0], y=given[0], z=given[1])
        else:
            # 计算形如H(X,Y|Z)的条件熵
            h_given = entropy(df, x=given[0])
            h_joint = joint_entropy(df, x=target[0], y=target[1], z=given[0])

    return h_joint - h_given


def mutual_info(df, var1, var2):
    """
    计算两个离散变量的互信息
    I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
    Parameters
    ----------
    df: pandas.DataFrame
    var1: str
        变量名1
    var2: str
        变量名2

    Returns
    -------
    float
    """
    h_self1 = entropy(df, x=var1)    # 变量1信息熵
    h_self2 = entropy(df, x=var2)    # 变量2信息熵
    h_joint = joint_entropy(df, x=var1, y=var2)    # 两个变量的联合信息熵

    return h_self1 + h_self2 - h_joint


def conditional_mutual_info(df, var, cond, target):
    """
    计算离散变量和target，在给定条件变量cond下的互信息
    I(var;target|cond) = H(target|cond) - H(target|var, cond) = H(var|cond) - H(var|cond, target)
    Parameters
    ----------
    df: pandas.DataFrame
    var: str
        变量名
    cond: str
        条件变量名
    target: str
        建模的target，此变量只允许服从0-1分布的变量

    Returns
    -------
    float
    """
    h_cond1 = conditional_entropy(df, target=var, given=cond)
    h_cond2 = conditional_entropy(df, target=var, given=[cond, target])

    return h_cond1 - h_cond2
