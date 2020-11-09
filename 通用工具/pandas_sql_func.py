# coding: utf-8
# Author: Jingcheng Qiu

"""
在DataFrame上实现一些SQL函数的功能
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset


def cut_timestamp(ts, unit):
    """
    截取timestamp，根据unit保留精度，eg. 2018-03-02 17:50:53 → 2018-03-02
    Parameters
    ----------
    ts: pandas.Series
        时间序列，dtype必须是str或timestamp
    unit: str, options:['year', 'month', 'day']
        时间单位

    Returns
    -------
    ts_new: pandas.Series
    """
    if unit == 'year':
        ts_new = ts.apply(lambda x: str(x)[:4])
    elif unit == 'month':
        ts_new = ts.apply(lambda x: str(x)[:7])
    elif unit == 'day':
        ts_new = ts.apply(lambda x: str(x)[:10])
    else:
        raise ValueError('Wrong unit!')

    return ts_new


def add_months(ts, month):
    """
    给日期加月份，等同于SQL中的add_months函数
    Parameters
    ----------
    ts: pandas.Series
        时间序列, dtype必须是str或timestamp
    month: int
        加的月份数, 正数加, 负数减

    Returns
    -------
    ts_new: pandas.Series
    """
    ts_new = pd.to_datetime(ts, errors='coerce')
    offset = DateOffset(months=abs(month))
    if month > 0:
        ts_new = ts_new.apply(lambda x: x + offset)
    elif month < 0:
        ts_new = ts_new.apply(lambda x: x - offset)
    else:
        raise ValueError('month can not equal 0!')

    return ts_new


def timediff(ts1, ts2, unit='day'):
    """
    计算两个日期的时间差，根据不同的计算单位返回结果
    Parameters
    ----------
    ts1: pandas.Series
        时间序列, dtype必须是str或timestamp
    ts2: pandas.Series
        时间序列, dtype必须是str或timestamp
    precision: str, options:['second', 'day', 'month']
        时间差的单位

    Returns
    -------
    timedelta: pandas.Series
        ts1与ts2的时间差
    """
    if ts1.shape[0] != ts2.shape[0]:
        raise ValueError('Difference shape between series1 and series2!')

    # 计算timestamp的差值
    try:
        timedelta = pd.to_datetime(ts1, errors='coerce') - pd.to_datetime(ts2, errors='coerce')
    except Exception as e:
        raise e

    # 根据计算精度返回结果
    if unit == 'second':
        return timedelta.apply(lambda x: x.total_seconds())   # 秒
    elif unit == 'day':
        return timedelta.apply(lambda x: x.days)              # 天
    elif unit == 'month':
        return timedelta.apply(lambda x: int(x.days/30.0))    # 月
    else:
        raise ValueError('Wrong calculating unit!')


def row_number(df, part_by, order_by, to_date=False, method='first', ascending=True):
    """
    对数据进行分组，再根据其他字段进行排序，返回序号, 等同于SQL中的ROW_NUMBER() OVER(PARTITION BY ... ORDER BY ...)函数
    Parametes
    ---------
    df: pandas.DataFrame
    part_by: str or list
        分组的列名
    order_by: str
        排序的列名
    to_date: bool, default False
        是否将变量转成datetime格式，如果用于排序的列是字符串格式的日期需填True
    method: str, options ['first', 'max', 'min', 'dense']
        排序方式
    ascending: bool, default True
        是否升序排列

    Return
    ------
    rank_num: pandas.Series
        每行的排序编号
    """
    if to_date:
        if isinstance(df[order_by].iloc[0], str):
            df[order_by] = pd.to_datetime(df[order_by], errors='coerce')
        elif type(df[order_by].iloc[0]) == pd._libs.tslib.Timestamp:
            print('"{}” is aready datetime type'.format(order_by))
        else:
            raise ValueError('invalid data type "{0}"'.format(order_by))
    group = df.groupby(part_by)
    rank_num = group[order_by].rank(method=method, ascending=ascending)

    return rank_num


def group_stat(df, group_vars, stat_var, stat_method, filter_vars=None, new_var=None):
    """
    按group_var列聚合计算calc_var列的stat_method统计值，输出聚合后的结果
    Parametes
    ---------
    df: pandas.DataFrame
    group_vars: str or list
        分组的列名
    stat_var: str
        计算统计信息的列名
    stat_method: str
        统计方式, 可选['avg', 'max', 'min', 'count', 'sum', 'unique count']
    filter_vars: str or list, default None
        筛选变量的列名, 该变量只能是取值为0、1的变量，默认为None
    new_var: str, default None
        新的变量名, 默认为None

    Return
    ------
    result: DataFrame or array, 聚合后的数据，若是DataFrame则包含两列，一列是聚合变量的值，另一列是统计值，
            若是array则是统计值
    """
    if type(group_vars) == str:
        group_vars = [group_vars]
    if type(filter_vars) == str:
        filter_vars = [filter_vars]

    # 复制一个临时的dataframe，将filter_var=0的stat_var的值变为nan，起到类似case when的效果
    if filter_vars != None and filter_vars != []:
        df_tmp = df[group_vars + [stat_var]].copy()
        for var in filter_vars:
            df_tmp.loc[df[var] == 0, stat_var] = np.nan
    else:
        df_tmp = df[group_vars + [stat_var]]

    group = df_tmp.groupby(group_vars)
    if stat_method == 'avg':
        result = group[stat_var].mean()
    elif stat_method == 'max':
        result = group[stat_var].max(skipna=True)
    elif stat_method == 'min':
        result = group[stat_var].min(skipna=True)
    elif stat_method == 'count':
        result = group[stat_var].count()
    elif stat_method == 'sum':
        result = group[stat_var].sum(skipna=True)
    elif stat_method == 'std':
        result = group[stat_var].std()
    elif stat_method == 'unique count':
        result = group[stat_var].unique().apply(lambda x: x.shape[0])
    else:
        raise ValueError('Wrong stat_method!')

    # 根据是否指定了new_var返回DataFrame或array
    if new_var != None:
        result = result.to_frame().reset_index(drop=False)
        result.columns = group_vars + [new_var]
        return result
    else:
        return result.values


def shift_calc(df, part_by, order_by, calc_var, calc_method, to_date=False):
    """
    根据order_by进行错位运算，用于算一些序列的差值或比值
    Parametes
    ---------
    df: pandas.DataFrame
    part_by: str
        分组的列名
    order_by: str
        排序的列名
    calc_var: str
        需要计算的列名
    calc_method: str, options: ['subtraction', 'division']
        计算方式, 可计算差值和比率

    Return
    ------
    result: pandas.Series
        错位运算的结果，第一条记录用0填充
    """
    df_tmp1 = row_number(df, part_by=part_by, order_by=order_by, to_date=to_date, method='first', ascending=True)
    df_tmp2 = df_tmp1[[part_by, calc_var, 'rownum_'+order_by]].copy()
    df_tmp2['rownum_'+order_by] = df_tmp1['rownum_'+order_by] + 1

    # 错位相连
    df_merge = pd.merge(df_tmp1, df_tmp2, how='left', on=[part_by, 'rownum_'+order_by], suffixes=['_1', '_2'])

    # 进行计算
    if calc_method == 'subtraction':
        result = df_merge[calc_var+'_1'] - df_merge[calc_var+'_2']
    elif calc_method == 'division':
        result = df_merge[calc_var+'_1'] / df_merge[calc_var+'_2']
    else:
        raise ValueError('Wrong calc_method!')

    # 对结果进行处理
    result = result.fillna(0)       # 第一条记录的结果用0填充
    result = result.replace([np.inf, -np.inf], 0)       # 将无穷大替换成0

    return result
