# coding: utf-8
# Author: Jingcheng Qiu

"""
一些日常使用的小工具
"""


import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler


def gene_date_list(start, days):
    """
    生成连续日期的列表
    Parameters
    ----------
    start: str
        开始日期, eg.'2018-09-21'
    days: int
        生成多少天的日期，正数递增生成，负数递减生成

    Return
    ------
    date_list: list
        日期列表
    """
    date_list = []
    start = datetime.strptime(start, '%Y-%m-%d')
    for i in range(0, days, int(days / abs(days))):
        date_list.append(datetime.strftime(start + timedelta(i), '%Y-%m-%d'))

    return date_list


def scheduler_job(main, execution_time):
    """
    定时执行main函数
    Parameters
    ----------
    main: function
        需要定时执行的main函数, 注意main函数不能带参数
    execution_time: str
        任务开始执行的时间, eg. '18:00'
    """
    if len(execution_time.split(':')) != 2:
        raise ValueError('invalid execution_time format')
    hour = int(execution_time.split(':')[0])
    minute = int(execution_time.split(':')[1])
    scheduler = BackgroundScheduler()
    scheduler.add_job(main, 'cron', hour=hour, minute=minute)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
    try:
        while True:
            time.sleep(300)    # 等待300秒重新检查时间
            print('sleep!')
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print('Exit The Job!')


def plot_polynomial(coefficient, power, low=0.0, up=1.0, freq=0.01):
    """
    绘制多项式函数图形
    Parameters
    ----------
    coefficient: numpy.ndarray
        多项式系数
    power: list
        系数的幂次排列，eg.[0,1,2,3]
    low: float
        定义域下界
    up: float
        定义域上界
    freq: float
        描点频率
    """
    if len(coefficient) != len(power):
        raise ValueError('coefficient and power must have same length')
    x = np.arange(low, up+freq, freq)
    mat = np.zeros((len(x), len(coefficient)))

    # 生成自变量的多项式项
    for i, p in enumerate(power):
        mat[:, i] = np.power(x, p)
    predict = np.dot(mat, coefficient)

    # 画图
    plt.figure(figsize=(4, 4), dpi=150)
    plt.plot(x, predict)
    plt.xlabel('score', fontsize=9)
    plt.ylabel('predict', fontsize=9)
    plt.show()
    plt.close()


def check_df_diff(df1, df2, precision=1e-4):
    """
    比较两个dataframe是否相同
    """
    if df1.shape != df2.shape:
        raise Exception('两个Dataframe大小不一致')
    if sum(df1.columns != df1.columns) > 0:
        raise Exception('两个数据框的列名不一致')

    col_types = df1.dtypes
    col_names = df1.columns
    check_col = []
    res = {}

    # 粗略比较，先筛选出均值或者中位数不相等的列
    for col in col_names:
        if col_types[col] in ('int32', 'int64', 'float32', 'float64'):
            if abs(df1[col].mean() - df2[col].mean()) > precision or abs(
                            df1[col].quantile(0.5) - df2[col].quantile(0.5)) > precision:
                check_col.append(col)
        elif col_types[col] == 'object':
            check_col.append(col)
        else:
            print('{0} is not a available type'.format(col))

    # 精确比较两个dataframe的每一个元素是否相等
    for col in check_col:
        diff_row = []
        for row in range(df1.shape[0]):
            if df1.loc[row, col] == df1.loc[row, col] or df2.loc[row, col] == df2.loc[row, col]:
                if col_types[col] in ('float32', 'float64'):
                    if abs(df1.loc[row, col] - df2.loc[row, col]) > precision:
                        diff_row.append(row)
                else:
                    if df1.loc[row, col] != df2.loc[row, col]:
                        diff_row.append(row)
        res[col] = diff_row

    return res, check_col
