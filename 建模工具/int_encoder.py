# coding: utf-8
# Author: Jingcheng Qiu

"""
正整数编码工具
"""


# from multiprocessing import Pool
# from itertools import repeat
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import sparse


def index_to_matrix(arr, colnums, to_sparse=True):
    """
    根据序列值生成0-1矩阵, 有序号的地方为1，其他为0
    Parameters
    ----------
    arr: numpy.ndarray
        正整数数组
    colnums: int
        矩阵的列数, 列数必须大于序号的最大值，eg. 序号的范围是[0, 5], columns需要填入大于等于6的整数
    to_sparse: bool, default True
        是否保存为稀疏矩阵

    Returns
    -------
    mat: scipy.sparse.csr_matrix or numpy.ndarray

    Examples
    --------
    >>> a = np.asarray([[0, 3, 1],
                        [3, 2, 0]])
    >>> mat = index_to_matrix(a, colnums=4, to_sparse=False)
    >>> mat
    array([[1, 1, 0, 1],
           [1, 0, 1, 1]])
    """
    if not arr.dtype.name.startswith('int'):
        raise ValueError('"arr" data type must be integer')
    if arr.min() < 0:
        raise ValueError('there are negative numbers in the array')
    if arr.max() >= colnums:
        raise ValueError('`colnums` must be greater than the maximum value of the array')

    rownums = arr.shape[0]
    indices = np.empty(0, dtype=int)    # 每行的起止索引
    indptr = np.zeros(rownums + 1, dtype=int)  # 非零元素索引
    for i, row in enumerate(arr):
        indices = np.append(indices, row)
        indptr[i + 1] = indptr[i] + len(row)

    values = np.ones(indptr[-1], dtype=float)
    mat = sparse.csr_matrix((values, indices, indptr), shape=(rownums, colnums), dtype=float)
    if to_sparse is False:
        mat = mat.toarray()

    return mat


def batch_idx_to_mat(arr, colnums, batch_size=1, verbose=-1, to_sparse=True):
    """
    分批生成0-1矩阵, 加快计算速度, 防止内存溢出
    Parametes
    ---------
    arr: numpy.ndarray
        正整数数组
    colnums: int
        矩阵的列数, 列数必须大于序号的最大值，eg. 序号的范围是[0, 5], columns需要填入大于等于6的整数
    batch_size: int, default 1
        分批数, 每轮计算150-200行时速度最快
    verbose: int, default -1
        每隔多少批显示一次进度, -1则不显示当前进度
    to_sparse: bool, default True
        是否保存为稀疏矩阵

    Returns
    -------
    mat: scipy.sparse.csr_matrix or numpy.ndarray

    Examples
    --------
    >>> a = np.asarray([[0, 3, 1],
                        [3, 2, 0],
                        [1, 0, 2],
                        [3, 0, 1])
    >>> mat = batch_idx_to_mat(a, colnums=4, batch_size=2, to_sparse=False)
    >>> mat
    array([[1., 1., 0., 1.],
           [0., 1., 1., 1.],
           [0., 1., 1., 1.],
           [0., 1., 1., 1.],
           [0., 1., 1., 1.]])
    """
    if isinstance(arr, pd.Series):
        arr = arr.values

    idx = np.linspace(0, arr.shape[0], batch_size + 1, dtype=int)    # 按行切片, 生成切片索引
    stack = []
    for b in range(idx.size - 1):
        if verbose > 0 and (b + 1) % verbose == 0:
            print('\r', '========================== Batch: {0}/{1} =========================='.format(b + 1, batch_size), end='', flush=True)
        indices = arr[idx[b]:idx[b + 1]]
        stack.append(index_to_matrix(indices, colnums=colnums, to_sparse=True))
    mat = sparse.vstack(stack)

    if to_sparse is False:
        mat = mat.toarray()

    return mat


# def multi_batch_onehot(arr, colnums, n_jobs=1, batch_size=1, save_sparse=True):
#     """
#     多线程分批生成one-hot矩阵(当行数较多时速度比batch_int2onehot慢)
#     Parameters
#     ----------
#     arr: numpy.ndarray
#         每个样本的appid list
#     colnums: int
#         one-hot矩阵的列数
#     n_jobs: int, default 1
#         进程数
#     batch_size: int, default 1
#         分批数
#     save_sparse: bool, default True
#         是否存为稀疏矩阵
#
#     Returns
#     -------
#     mat: sparse.csr_matrix or numpy.ndarray
#         one-hot矩阵
#     """
#     if isinstance(arr, pd.Series):
#         arr = arr.values
#
#     # slice
#     idx = np.linspace(0, arr.shape[0], n_jobs + 1, dtype=int)
#     queue = []
#     for b in range(idx.size - 1):
#         queue.append(arr[idx[b]:idx[b + 1]])
#
#     # multiprocess
#     pool = Pool(n_jobs)
#     res = pool.starmap(int2onehot, zip(queue, repeat(colnums), repeat(batch_size)))
#     pool.close()
#     pool.join()
#     mat = sparse.vstack(res)
#     if save_sparse is False:
#         mat = mat.toarray()
#
#     return mat


def int_onehot_encoder(arr, colnum, to_sparse=True):
    """
    对整数数组进行One-hot编码
    Parameters
    ----------
    arr: numpy.ndarray
        一维整数数组
    colnum: int
        One-hot后的列数，取值必须大于数组中的最大值, eg. 数组的取值范围是[0, 5], 需要填入大于等于6的整数
    to_sparse: bool, default True
        是否保存为稀疏矩阵

    Returns
    -------
    onehot_arr: scipy.sparse.csr_matrix
        One-hot稀疏矩阵

    Examples
    --------
    >>> a = np.asarray([0, 3, 1, 2, 2])
    >>> onehot_arr = int_onehot_encoder(a, colnums=4, to_sparse=False)
    >>> onehot_arr
    array([[1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 1., 0.]])
    """
    if arr.ndim != 1:
        raise ValueError('`arr` dimension must be 1')
    if not arr.dtype.name.startswith('int'):
        raise ValueError('`arr` data type must be integer')
    if arr.min() < 0:
        raise ValueError('there are negative numbers in the array')
    if arr.max() >= colnum:
        raise ValueError('`colnum` must be greater than the maximum value of the array')

    res = np.eye(colnum, colnum)[arr]
    if to_sparse:
        res = sparse.csr_matrix(res, shape=(arr.size, colnum), dtype=float)

    return res
