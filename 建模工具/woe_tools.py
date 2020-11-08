# coding: utf-8
# Author: Jingcheng Qiu
# Version: 2.2

"""
变量WOE工具包，包含变量分箱、计算WOE和IV、WOE值替换
"""

__all__ = ['init_binning',
           'plot_reference',
           'numwoe_autobinning',
           'numwoe_aptbinning',
           'catwoe_autobinning',
           'catwoe_isobinning',
           'catwoe_aptbinning',
           'catwoe_apply',
           'numwoe_apply',
           'iv_extract',
           'replace_var_woe_ref']


import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import tree
import contingency as cy
import matplotlib.pyplot as plt
from plot_chs_config import zhfont


def woe_calc(bad, good, badsum, goodsum):
    """
    计算woe值
    Parameters
    ----------
    bad: int or float
        单个bin中正样本数
    good: int or float
        单个bin中负样本数
    badsum: int or float
        所有样本中正样本数
    goodsum: int or float
        所有样本中负样本数

    Returns
    -------
    float
    """
    if bad == 0:
        bad += badsum / float(goodsum)
        badsum += badsum / float(goodsum)
    if good == 0:
        good += goodsum / float(badsum)
        goodsum += goodsum / float(badsum)
    badprop = bad / float(badsum)
    goodprop = good / float(goodsum)

    return np.log(badprop / goodprop)


def iv_calc(count_1, count_0):
    """
    计算变量的iv值
    Parameters
    ----------
    count_1: pandas.Series or numpy.ndarray
        每个bin中正样本的数量
    count_0: pandas.Series or numpy.ndarray
        每个bin中负样本的数量

    Returns
    -------
    float
    """
    bad_dist = count_1 / float(count_1.sum())
    good_dist = count_0 / float(count_0.sum())
    bad_dist = bad_dist.apply(lambda x: 0.0001 if x == 0 else x)    # Laplace平滑, 防止出现0
    good_dist = good_dist.apply(lambda x: 0.0001 if x == 0 else x)

    return ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()


def init_binning(df, var_name, label, max_bin_num=200, missing=False, cut_points=None):
    """
    对数值型变量进行初试切bin，返回每个bin中分类标签的分布和bin的区间
    Parameters
    ----------
    df: pandas.DataFrame
        需包含变量列和标签列
    var_name: str
        变量名
    label: str
        分类标签列名
    max_bin_num: int
        切bin的最大数量，默认为200个
    missing: bool, default False
        变量是否有缺失
    cut_points: list or array-like
        指定的切分点，若不指定则根据分位点分bin

    Returns
    -------
    ds: pandas.DataFrame
        每个bin中样本的统计信息
    """
    df_tmp = df[[var_name, label]].copy()
    concatflag = False    # 用于记录是否需要合并缺失值的bin

    if missing:
        if df_tmp[var_name].isnull().sum() > 0:
            df_missing = df_tmp.loc[df_tmp[var_name].isnull(), :].reset_index(drop=True)
            df_tmp = df_tmp.loc[~df_tmp[var_name].isnull(), :].reset_index(drop=True)
            concatflag = True
        else:
            logging.warning('Expect variable has missing value but actually no missing')
    else:
        if df_tmp[var_name].isnull().sum() > 0:
            raise ValueError('"{0}" exist NaN'.format(var_name))

    # 初始化切分点
    if cut_points is not None:
        if len(cut_points) <= 1:
            raise ValueError('invalid cut points: {0}'.format(var_name))
        if np.max(df_tmp[var_name]) >= cut_points[-1]:
            cut_points[-1] = np.inf                   # bins最后一个值改为inf, 防止忘记填最大值
        if np.min(df_tmp[var_name]) < cut_points[0]:
            cut_points[0] = np.min(df_tmp[var_name])  # bins第一个值改为min value, 防止忘记填最小值

    # 若变量的unique value <= max_bin_num，则每个value一个bin
    elif len(df_tmp[var_name].unique()) <= max_bin_num:
        cut_points = np.sort(df_tmp[var_name].unique())
        cut_points = np.append(cut_points, np.inf)

    # 若变量的unique value > max_bin_num, 则根据分位点切max_bin_num个bin
    else:
        pct = np.arange(max_bin_num + 1) / float(max_bin_num)
        cut_points = df_tmp[var_name].quantile(pct, interpolation='higher').unique()  # 计算分位点并去重
        cut_points = cut_points.astype(float)    # 将切分点都改成，防止无法将最后一个值改成inf时报错
        cut_points[-1] = np.inf

    # 按切分点分bin，并计算每个bin中label的分布
    df_tmp[var_name + '_bin'] = np.digitize(df_tmp[var_name], bins=cut_points, right=False)
    ds = df_tmp.groupby(var_name + '_bin')[label].value_counts().unstack().fillna(value=0)
    ds['total'] = ds[0] + ds[1]
    ds['bin'] = [[cut_points[i-1], cut_points[i]] for i in list(ds.index)]
    ds['bin_lb'] = [cut_points[i-1] for i in list(ds.index)]
    ds = ds.sort_values(by='bin_lb', axis=0, ascending=True).reset_index(drop=True)  # 根据bin的下界进行排序
    ds.columns = ['0', '1', 'total', 'bin', 'bin_lb']

    if concatflag:
        count_1 = df_missing[label].sum()
        count_0 = df_missing.shape[0] - count_1
        ds_miss = pd.DataFrame([{'0': count_0, '1': count_1, 'total': df_missing.shape[0], 'bin': [np.nan]}])
        ds = pd.concat([ds_miss, ds], ignore_index=True)

    return ds


def __value_match(map_dict, key_list):
    """
    根据categorical变量的map_dict对index对应的值做匹配 eg:[1,3,4,5] → ['本科','大专','高中','研究生']
    Parameters
    ----------
    map_dict: dict
    key_list: list

    Returns
    -------
    result: list
    """
    result = []
    for key in key_list:
        if key in map_dict:
            result.append(map_dict[key])
        else:
            result.append('base')

    return result


def __mergebin(ds, idx_list, idx, var_type):
    """
    合并相邻的bin，并重新计算合并后bin的统计信息和卡方值
    Parameters
    ----------
    ds: pandas.DataFrame
    idx_list: list
    idx: int
        待合并的bin在index list中的位置
    var_type: str, options ['numerical', 'categorical']
        变量类型

    Returns
    -------
        ds: pandas.DataFrame
    """
    # 合并两个bin，重新计算bin的0-1分布
    ds.at[idx_list[idx], ['0', '1']] = ds.loc[idx_list[idx], ['0', '1']] + ds.loc[idx_list[idx+1], ['0', '1']]
    ds.at[idx_list[idx], 'total'] = ds.loc[idx_list[idx], 'total'] + ds.loc[idx_list[idx+1], 'total']
    # 重新计算bin的范围
    if var_type == 'numerical':
        ds.at[idx_list[idx], 'bin'] = [ds.loc[idx_list[idx], 'bin'][0], ds.loc[idx_list[idx+1], 'bin'][1]]
    elif var_type == 'categorical':
        ds.at[idx_list[idx], 'bin'] = ds.loc[idx_list[idx:idx+2], 'bin'].sum()
    ds = ds.drop(idx_list[idx+1], axis=0)        # 删除被合并的bin
    idx_list.pop(idx+1)      # 从index list中删除被合并的bin的index

    # 重新计算合并后的bin与前后两个bin的卡方值
    # 若不是第一个bin，则计算与前一个bin的卡方值
    if idx != 0:
        ds.at[idx_list[idx-1], 'chisq'] = cy.chi2_contingency(ds.loc[idx_list[(idx-1):(idx+1)], ['0', '1']])[0]
    # 若不是最后一个bin，则计算与后一个bin的卡方值，否则卡方值赋值成无穷大
    if idx < ds.shape[0] - 1:
        ds.at[idx_list[idx], 'chisq'] = cy.chi2_contingency(ds.loc[idx_list[idx:idx+2], ['0', '1']])[0]
    else:
        ds.at[idx_list[idx], 'chisq'] = 9999999.0

    return ds


def __gene_reference(ds, var_name, var_type):
    """
    生成变量的woe reference table
    Parameters
    ----------
    ds: pandas.DataFrame
    var_name: str
        变量名
    var_type: str, options ['numerical', 'categorical']
        变量类型

    Returns
    -------
    df_ref_table: pandas.DataFrame
        reference table
    """
    # 计算每个bin的woe和iv值
    goodsum = ds['0'].sum()
    badsum = ds['1'].sum()
    ds['woe_value'] = ds.apply(lambda x: woe_calc(x['1'], x['0'], badsum, goodsum), axis=1)
    iv = iv_calc(ds['1'], ds['0'])

    # 生成reference table
    df_ref_table = pd.DataFrame(columns=['Var_Name', 'Var_Type', 'Bin_No', 'Var_Value', 'Ref_Value',
                                         'Count_0', 'Count_1', 'Total', 'Bad_Rate', 'Proportion', 'IV'])
    df_ref_table['Bin_No'] = range(1, ds.shape[0] + 1)   # Bin的编号，从1开始
    df_ref_table['Var_Value'] = ds['bin'].astype(str)    # 将list转成字符串
    df_ref_table['Ref_Value'] = ds['woe_value']
    df_ref_table['Count_0'] = ds['0']
    df_ref_table['Count_1'] = ds['1']
    df_ref_table['Total'] = ds['total']
    df_ref_table['Bad_Rate'] = 1.0 * df_ref_table['Count_1'] / df_ref_table['Total']
    df_ref_table['Proportion'] = 1.0 * df_ref_table['Total'] / ds['total'].sum()
    df_ref_table['IV'] = iv
    df_ref_table['Var_Name'] = var_name
    df_ref_table['Var_Type'] = var_type

    return df_ref_table


def __get_list_str(x):
    """
    取categorical变量的取值, 三个为一排排列
    """
    str_list = x.split('\001')

    # 如果这个bin的取值太多，则返回省略号
    if len(str_list) > 10:
        return '...'

    res = ''
    for i in range(len(str_list)):
        res += str_list[i] + ','
        if (i + 1) % 3 == 0 and i + 1 != len(str_list):
            res += '\n'

    return res[:-1]


def plot_reference(df_ref, title=None, to_show=True, save_path=None):
    """
    根据WOE reference table绘制WOE图
    Parameters
    ----------
    df_ref: pandas.DataFrame
        WOE reference table
    title: str, default None
        图片标题
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存放路径
    """
    x = np.arange(df_ref.shape[0])
    y = df_ref['Ref_Value'].values
    z = df_ref['Bad_Rate'].values
    var_name = df_ref['Var_Name'].iloc[0]
    if title is None:
        title = var_name
    iv = round(df_ref['IV'].iloc[0], 5)
    overall_dp = (df_ref['Bad_Rate'] * df_ref['Proportion']).sum()  # 整体的逾期率

    plt.figure(figsize=(8, 4), dpi=200)
    plt.bar(x, df_ref['Proportion'], color='royalblue', label='0', align='edge', width=0.985)
    plt.bar(x, df_ref['Proportion'] * df_ref['Bad_Rate'], color='firebrick', label='1', align='edge', width=0.985)

    # 绘制横坐标的label
    if df_ref['Var_Type'].iloc[0] == 'numerical':
        xticks_list = [tuple([float(j) for j in i.strip('([] ').split(',')]) for i in df_ref['Var_Value'].tolist()]
        xticks_list = [round(i[0], 4) for i in xticks_list]
        plt.xticks(x, xticks_list, fontsize=8)
    if df_ref['Var_Type'].iloc[0] == 'categorical':
        xticks_list = df_ref['Var_Value'].apply(__get_list_str).tolist()
        plt.xticks(x+0.5, xticks_list, fontproperties=zhfont, fontsize=8)
    plt.axis(ymin=0.0, ymax=1.0)
    plt.yticks(fontsize=8)
    plt.ylabel('proportion', fontsize=8)
    plt.legend(loc=2, fontsize=8)
    ax2 = plt.twinx()
    plt.plot(x+0.5, y, color='black', linewidth=1.5)
    for i, j, k in zip(x, y, z):
        ax2.annotate('%.2f(%.2f%%)' % (j, k*100), xy=(i+0.5, j), va='center', ha='center',
                     bbox={'boxstyle': 'round', 'fc': 'w'}, fontsize=8)
    plt.axhline(0.0, color='dimgrey', linestyle='dashed', linewidth=0.8, label='Overall bad rate')
    ax2.annotate('Overall bad rate: {}%'.format(round(100*overall_dp, 2)), color='r',
                 xy=(0.0, 0.0), xytext=(-0.3, 0.01), fontsize=6)
    plt.yticks(fontsize=8)
    plt.ylabel('Woe value(Bad rate)', fontsize=8)
    plt.title('{0}: IV={1}'.format(title, iv), fontproperties=zhfont, fontsize=12)

    # 保存图片
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '{0}.png'.format(var_name)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()


def numwoe_autobinning(df, var_name, label, max_bins=6, min_prop_in_bin=0.05, missing=True, max_bin_init=200,
                       method='chisq', to_plot=True, to_show=True, title=None, save_path=None):
    """
    对numerical变量进行自动分bin并计算woe和iv值，可选卡方值和信息熵的方式对变量进行分bin
    Parameters
    -----------
    df: pandas.DataFrame
    var_name: str
        变量名
    label: str
        分类标签列名
    max_bins: int, defulat 6
        最大分bin个数
    min_prop_in_bin: float, default 0.05
        每个bin中的最小样本数量占比
    missing: bool, default True
        是否有缺失值
    max_bin_init: int, defulat 200
        初始分bin的最大分bin个数，一般根据样本数量和unique value来决定
    method: str, default 'chisq', options ['chisq', 'entropy']
        分bin的方式
    to_plot: bool, default True
        是否绘制reference图
    title: str, default None
        图片标题
    to_show: bool, default True
        是否展示图片
    save_path: str
        图片存放的文件夹路径

    Returns
    --------
    df_ref_table: pandas.DataFrame
        woe reference table
    """
    min_samples_in_bin = int(df.shape[0] * min_prop_in_bin)    # 计算每个bin的最小样本个数

    if method == 'chisq':
        ds = init_binning(df, var_name=var_name, label=label, max_bin_num=max_bin_init, missing=missing)
        # 计算相邻两个bin的卡方值
        chisq = []
        for i in range(ds.shape[0] - 1):
            chisq.append(cy.chi2_contingency(ds.iloc[[i, i + 1], [0, 1]])[0])
        chisq.append(9999999.0)
        ds['chisq'] = chisq

        # 把missing值单独作为一个bin
        if missing:
            if ds['bin_lb'].isnull().sum() > 0:
                ds_miss = ds.loc[ds['bin_lb'].isnull(), :]
                ds = ds.loc[~ds['bin_lb'].isnull(), :]
            else:
                ds_miss = pd.DataFrame()

        # 合并卡方值较小的两个bin
        ds_idx_list = list(ds.index)
        while (ds.shape[0] > max_bins) | (ds['chisq'].min() <= stats.chi2.ppf(0.95, 1)):
            # 找到卡方值最小的bin的index在index list中的位置
            k = ds_idx_list.index(ds['chisq'].idxmin())
            ds = __mergebin(ds, idx_list=ds_idx_list, idx=k, var_type='numerical')

        # 限制每个bin的最小样本个数
        while (ds['total'].min() < min_samples_in_bin) & (ds.shape[0] > 2):
            # 找到样本个数最少的bin
            k = ds_idx_list.index(ds['total'].idxmin())
            # 如果与前一个bin的卡方值比后一个bin的卡方值小，则选择与前一个bin合并
            if (k == len(ds_idx_list) - 1) | (ds.loc[ds_idx_list[k], 'chisq'] > ds.loc[ds_idx_list[k - 1], 'chisq']):
                k -= 1
            ds = __mergebin(ds, idx_list=ds_idx_list, idx=k, var_type='numerical')

    elif method == 'entropy':
        max_depth = int(np.log(max_bins) / np.log(2.0))
        if missing:
            df_miss = df.loc[df[var_name] == -1.0, [var_name, label]].reset_index(drop=True)
            df_no_miss = df.loc[df[var_name] != -1.0, [var_name, label]].reset_index(drop=True)
            if df_miss.shape[0] > 0:
                ds_miss = init_binning(df_miss, var_name=var_name, label=label, cut_points=[-1.0, -0.5], missing=False)
            else:
                ds_miss = pd.DataFrame()
            min_value = -0.5
        else:
            df_no_miss = df
            min_value = df_no_miss[var_name].min()
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_in_bin)
        clf.fit(df_no_miss[var_name].values.reshape(-1, 1), df_no_miss[label])
        cut_points = np.sort(clf.tree_.threshold[clf.tree_.threshold != -2.0])
        cut_points = np.append([min_value], cut_points)
        cut_points = np.append(cut_points, df_no_miss[var_name].max())
        ds = init_binning(df_no_miss, var_name=var_name, label=label, cut_points=cut_points, missing=False)

    else:
        raise ValueError('method can only be "chisq" or "entropy"')

    # 生成最终的reference table
    if missing:
        ds = pd.concat([ds_miss, ds])
    ds = ds.reset_index(drop=True)
    df_ref_table = __gene_reference(ds, var_name=var_name, var_type='numerical')

    # 画图
    if to_plot:
        plot_reference(df_ref_table, title=title, to_show=to_show, save_path=save_path)

    return df_ref_table


def numwoe_aptbinning(df, var_name, label, bins, missing=True, to_plot=True, to_show=True, title=None, save_path=None):
    """
    根据指定的切分区间对数值型变量进行分箱并计算woe和iv值
    Parameters
    -----------
    df: pandas.DataFrame
    var_name: str
        变量名
    label: str
        分类标签列名
    bins: list or array-like
        指定的切分点，eg. [0.0, 1.0, 3.0, 10.0]
    missing: bool, default True
        是否有缺失值
    to_plot: bool, default True
        是否绘制reference图
    title: str, default None
        图片标题
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存放的文件夹路径

    Returns
    -------
    df_ref_table: pandas.DataFrame
        woe reference table
    """
    ds = init_binning(df, var_name=var_name, label=label, cut_points=bins, missing=missing)
    df_ref_table = __gene_reference(ds, var_name=var_name, var_type='numerical')
    if to_plot:
        plot_reference(df_ref_table, title=title, to_show=to_show, save_path=save_path)

    return df_ref_table


def catwoe_autobinning(df, var_name, label, max_bins=6, min_prop_in_bin=0.05, min_samples_init=1,
                       missing_value=None, to_plot=True, to_show=True, title=None, save_path=None):
    """
    对categorical变量进行自动分箱并计算woe和iv值
    Parameters
    ----------
    df: pandas.DataFrame
    var_name: str
        变量名
    label: str
        分类标签列名
    max_bins: int
        最大分bin个数, 默认为6个
    min_samples_init: int
        初始分bin时的最小样本个数，将样本数较少的变量取值合并到一个bin中，默认为1
    min_prop_in_bin: float, default None
        每个bin中的最小样本数量占比
    missing_value: str, default None
        指定缺失值，分bin时会把该取值单独分到一个bin
    to_plot: bool, default True
        是否绘制reference图
    to_show: bool, default True
        是否展示图片
    title: str, default None
        图片标题
    save_path: str, default None
        图片存放路径

    Returns
    -------
    df_ref_table: pandas.DataFrame
        woe reference table
    """
    min_samples_in_bin = int(df.shape[0] * min_prop_in_bin)
    ds = pd.crosstab(df[var_name], df[label]).fillna(value=0).reset_index(drop=False)
    ds[var_name] = ds[var_name].astype(str)
    ds['total'] = ds[1] + ds[0]
    ds['bin'] = [[i] for i in ds.index]  # 给每个bin编号
    ds.columns = ['value', '0', '1', 'total', 'bin']
    map_dict = dict(zip(ds.index, ds['value']))  # 生成每个编号与真实取值的映射

    # 把缺失值单独作为一个bin
    if missing_value is not None:
        ds_miss = ds[ds['value'] == missing_value].copy()
        ds = ds[ds['value'] != missing_value]

    ds = ds.sort_values(by=['total'], ascending=True)  # 根据每个bin中的样本个数排序

    # 合并样本数量过小的bin
    idx_small_bin = list(ds[ds['total'] < min_samples_init].index)
    if len(idx_small_bin) >= 2:
        ds.at[idx_small_bin[0], ['0', '1']] = ds.loc[idx_small_bin, ['0', '1']].sum()
        ds.at[idx_small_bin[0], 'total'] = ds.loc[idx_small_bin, 'total'].sum()
        ds.at[idx_small_bin[0], 'bin'] = idx_small_bin
        ds = ds.drop(idx_small_bin[1:], axis=0)

    # 计算每个bin的bad rate
    ds['bad_rate'] = ds['1'] / (ds['0'] + ds['1'])
    ds = ds.sort_values(by='bad_rate', ascending=True)  # 根据每个bin中的bad rate排序

    # 计算相邻两个bin的卡方值
    chisq = []
    for i in range(ds.shape[0] - 1):
        chisq.append(cy.chi2_contingency(ds.iloc[[i, i + 1], [1, 2]])[0])
    chisq.append(9999999.0)
    ds['chisq'] = chisq

    # 循环合并相邻两个bin，重新计算与前后两个bin的卡方值
    ds_idx_list = list(ds.index)
    while (ds.shape[0] > max_bins) | (ds.chisq.min() <= stats.chi2.ppf(0.95, 1)):
        k = ds_idx_list.index(ds['chisq'].idxmin())
        ds = __mergebin(ds, idx_list=ds_idx_list, idx=k, var_type='categorical')

    # 限制每个bin的最小样本数
    while (ds['total'].min() < min_samples_in_bin) & (ds.shape[0] > 2):
        # 找到样本个数最少的bin
        k = ds_idx_list.index(ds['total'].idxmin())
        if (k == len(ds_idx_list) - 1) | (ds.loc[ds_idx_list[k], 'chisq'] > ds.loc[ds_idx_list[k - 1], 'chisq']):
            k -= 1
        ds = __mergebin(ds, idx_list=ds_idx_list, idx=k, var_type='categorical')

    # 生成reference table
    if missing_value is not None:
        ds = pd.concat([ds_miss, ds])
    ds = ds.reset_index(drop=True)
    ds['bin'] = ds['bin'].apply(lambda x: __value_match(map_dict, x))  # 将索引还原成变量原本的取值
    ds['bin'] = ds['bin'].apply(lambda x: '\001'.join(x))  # 用特殊符号'\001'拼接value，防止出现value中有标点符号
    df_ref_table = __gene_reference(ds, var_name=var_name, var_type='categorical')

    # 画图
    if to_plot:
        plot_reference(df_ref_table, title=title, to_show=to_show, save_path=save_path)

    return df_ref_table


def catwoe_aptbinning(df, var_name, label, bins, to_plot=True, to_show=True, title=None, save_path=None):
    """
    根据指定的分组对categorical变量进行分箱并计算WOE
    Parameters
    ----------
    df: pandas.DataFrame
    var_name: str
        变量名
    label: str
        分类标签列名
    bins: list
        分组规则，eg. [['初中', '高中'], ['大专', '本科', '硕士研究生'], ['博士研究生']]
    to_plot: bool, default True
        是否绘制reference图
    to_show: bool, default True
        是否展示图片
    title: str, default None
        图片标题
    save_path: str, default None
        图片存放路径

    Returns
    -------
    df_ref_table: pandas.DataFrame
        woe reference table
    """
    unique_values = set(sum(bins, []))
    if len(unique_values) != len(sum(bins, [])):
        raise ValueError('Value is repetitive, please check bins is correct')

    ds = pd.crosstab(df[var_name], df[label]).fillna(value=0).reset_index(drop=False)
    ds[var_name] = ds[var_name].astype(str)
    ds['total'] = ds[1] + ds[0]
    ds.columns = ['bin', '0', '1', 'total']

    # 根据指定的分bin方式分bin
    res = []
    for b in bins:
        idx_list = []
        for value in b:
            if value in ds['bin'].tolist():
                idx_list.append(int(ds[ds['bin'] == value].index.values))
        ds.at[idx_list[0], ['0', '1']] = ds.loc[idx_list, ['0', '1']].sum()
        ds.at[idx_list[0], 'total'] = ds.loc[idx_list, 'total'].sum()
        ds.at[idx_list[0], 'bin'] = b
        res.append(ds.loc[idx_list[0], :].tolist())

    # 生成reference table
    res = pd.DataFrame(res, columns=['bin', '0', '1', 'total'])
    res['bin'] = res['bin'].apply(lambda x: '\001'.join(x))  # 用特殊符号'\001'拼接value，防止出现value中有标点符号
    df_ref_table = __gene_reference(res, var_name=var_name, var_type='categorical')

    # 画图
    if to_plot:
        plot_reference(df_ref_table, title=title, to_show=to_show, save_path=save_path)

    return df_ref_table



def catwoe_isobinning(df, var_name, label, to_plot=True, to_show=True, title=None, save_path=None):
    """
    将categorical变量的每个取值单独分进一个bin中，计算WOE和IV
    Parameters
    ----------
    df: pandas.DataFrame
    var_name: str
        变量名
    label: str
        label列名
    to_plot: bool, default True
        是否绘制reference图
    to_show: bool, default True
        是否展示图片
    title: str, default None
        图片标题
    save_path: str, default None
        图片存放路径

    Returns
    -------
    df_ref_table: pandas.DataFrame
        woe reference table
    """
    # 每个value单独一个bin
    ds = pd.crosstab(df[var_name], df[label]).fillna(value=0).reset_index(drop=False)
    ds[var_name] = ds[var_name].astype(str)
    ds['total'] = ds[1] + ds[0]
    ds.columns = ['bin', '0', '1', 'total']

    df_ref_table = __gene_reference(ds, var_name=var_name, var_type='categorical')
    if to_plot:
        plot_reference(df_ref_table, title=title, to_show=to_show, save_path=save_path)

    return df_ref_table


def __str_convert(x):
    """
    统一unicode和str类型
    """
    if type(x) in [int, float, np.float64]:
        return str(int(x))
    elif type(x) is str:
        return x
    elif type(x) is unicode:
        return x.encode('utf8')
    else:
        return x


def __restore_list(string, value_type):
    """
    将字符串还原成list，eg:'[1, 2]' → [1,2]
    Parameters
    ----------
    string: str
        需要还原的字符串
    var_type: str, options ['numerical', 'categorical']
        值的类型
    """
    if value_type == 'numerical':
        return [np.float(i.strip('[] ')) for i in string.split(',')]
    elif value_type == 'categorical':
        return string.split('\001')
    else:
        raise ValueError('Wrong value type')


def __cvlookup(value, map_dict):
    """
    查找变量取值对应的woe值
    Parameters
    ----------
    value: str
        变量取值
    map_dict: dict
        变量的woe值

    Returns
    -------
    woe_value: float
        woe值
    """
    if value in map_dict.keys():
        woe_value = map_dict[value]
    else:
        woe_value = 0

    return woe_value


def numwoe_apply(df, ref_table, var_name, overwrite=False, prefix='nwoe_'):
    """
    对单个numerical变量进行woe替换，替换方式为：若满足bin_lb[i] ≤ X ＜ bin_ub[i], 则将X替换成ref_value[i]
    Parameters
    ----------
    df: pandas.DataFrame
    ref_table: pandas.DataFrame
        WOE reference table
    var_name: str
        变量名
    overwrite: bool, default False
        是否覆盖原始变量的值
    prefix: str, default 'nwoe_'
        woe后变量的前缀, overwrite=True时此参数不起作用
    """
    if ref_table['Var_Name'].unique().tolist() != [var_name]:
        raise ValueError('ref table not match var_name')
    interval = ref_table['Var_Value'].apply(lambda x: __restore_list(x, 'numerical')).values  # 还原list
    bin_lb = np.asarray([i[0] for i in interval])    # bin的下界
    if np.isnan(bin_lb[0]):
        bin_lb = bin_lb[1:]
        missing_woe = ref_table['Ref_Value'].iloc[0]    # 缺失值对应的woe值
        base = ref_table['Ref_Value'].iloc[1]
        w = ref_table['Ref_Value'].iloc[1:].diff(periods=1).fillna(base).values  # 进行1阶差分
    else:
        missing_woe = 0.0
        base = ref_table['Ref_Value'].iloc[0]
        w = ref_table['Ref_Value'].diff(periods=1).fillna(base).values  # 进行1阶差分
    bin_lb[0] = -np.Infinity     # 将第一个bin的下界替换成负无穷
    bin_lb = bin_lb.reshape((1, -1))

    x = (df[var_name].values.reshape((-1, 1)) >= bin_lb) * 1.0
    res = np.dot(x, w) + missing_woe * df[var_name].isnull().values
    if overwrite:
        df[var_name] = res
    else:
        df[prefix + var_name] = res


def catwoe_apply(df, ref_table, var_name, overwrite=False, prefix='cwoe_'):
    """
    对单个categorical变量进行woe替换
    Parameters
    ----------
    df: pandas.DataFrame
    ref_table: pandas.DataFrame
        WOE reference table
    var_name: str
        变量名
    overwrite: bool, default False
        是否覆盖原始变量的值
    prefix: str, default 'cwoe_'
        woe后变量的前缀, overwrite=True时此参数不起作用
    """
    if ref_table['Var_Name'].unique().tolist() != [var_name]:
        raise ValueError('ref table not match var_name')
    var_value = ref_table['Var_Value'].apply(lambda x: __restore_list(x, 'categorical')).values  # 还原list
    tmp_value = df[var_name].apply(lambda x: __str_convert(x))  # 统一字符编码

    # 构造变量取值的字典
    value_list = []
    ref_value_list = []
    for i, lst in enumerate(var_value):
        value_list += lst
        ref_value_list += [ref_table['Ref_Value'].iloc[i]] * len(lst)
    value_dict = dict(zip(value_list, ref_value_list))
    if overwrite:
        df[var_name] = tmp_value.apply(lambda x: __cvlookup(x, value_dict))
    else:
        df[prefix + var_name] = tmp_value.apply(lambda x: __cvlookup(x, value_dict))


def iv_extract(woe_ref, sort=True, save_path=None):
    """
    根据WOE reference table提取每个变量的IV值，并降序排列
    Parameters
    ----------
    woe_ref: pandas.DataFrame
        WOE的reference table
    sort: bool
        是否按IV值降序排列
    save_path: str, default None
        csv存放路径

    Returns
    -------
    df_iv: DataFrame
    """
    iv = []
    for var in woe_ref['Var_Name'].unique():
        iv.append([var, woe_ref['IV'][woe_ref['Var_Name'] == var].iloc[0]])
    df_iv = pd.DataFrame(iv, columns=['Var_Name', 'IV'])
    if sort:
        df_iv = df_iv.sort_values(by='IV', ascending=False)
    if save_path is not None:
        df_iv.to_csv(save_path, index=False)

    return df_iv


def replace_var_woe_ref(woe_ref_all, woe_ref_var):
    """
    将woe_ref_all中某个变量的WOE结果替换成woe_ref_var
    Parameters
    ----------
    woe_ref_all: pandas.DataFrame
        所有变量的reference table
    woe_ref_var: pandas.DataFrame
        单个变量的reference table

    Returns
    -------
    woe_ref_new: pandas.DataFrame
        新的reference table
    """
    replace_var = woe_ref_var['Var_Name'].unique()[0]
    woe_ref_new = woe_ref_all.loc[woe_ref_all['Var_Name'] != replace_var, :].copy()
    woe_ref_new = pd.concat([woe_ref_new, woe_ref_var], axis=0, ignore_index=True)

    return woe_ref_new

def calc_psi(delp, valid):
    """
    计算模型的PSI指标
    Attributes
    ----------
    delp: numpy.ndarray
        新数据每个bin中样本的占比
    valid: numpy.ndarray
        建模数据每个bin中样本的占比，默认为0.1
    
    Returns
    -------
    psi: Float
    """
    if len(delp) != len(valid):
        raise ValueError('Different shape of arguments!')
    psi = np.sum((delp - valid) * np.log(delp / valid))

    return psi


class WoeStability(object):
    """
    变量WOE稳定性检查, 检查WOE分箱后每个bin的占比和WOE值在不同数据集(一般是不同时间段的样本)中的变化程度
    Attributes
    ----------
    psi_tend: DataFrame
        不同数据集对应的变量psi
    woe_tend: DataFrame
        不同数据集对应的变量woe

    Method
    ------
    calc_psi_by_period:
    """
    def __init__(self, variables, woe_ref, label, datanames):
        """
        Parameters
        ----------
        variables: dict, eg. {'numerical': ['age', 'amount'], 'categorical': ['sex', 'city']}
            需要检查稳定性的变量列表，分numerical和categorical两类
        woe_ref: pandas.DataFrame
            建模时生成的WOE reference表
        label: str
            分类标签列名
        datanames: list
            数据集的名称
        """
        self.variables = variables
        if 'numerical' not in self.variables:
            self.variables.update({'numerical': []})
        if 'categorical' not in self.variables:
            self.variables.update({'categorical': []})
        self.woe_ref = woe_ref
        self.label = label
        self.datanames = datanames
        self.psi_tend = None
        self.woe_tend = None

    def __group_stat(self, df, groupvar):
        groups = df.groupby(groupvar)
        res = groups[self.label].value_counts().unstack().sort_index().fillna(0)
        res['ratio_0'] = res[0] / res[0].sum()
        res['ratio_1'] = res[1] / res[1].sum()
        res['total'] = res[0] + res[1]
        res['proportion'] = res['total'] / df_part.shape[0]
        res['bad_rate'] = res[1] / res['total']
        res['new_ref_value'] = res.apply(lambda x: woe_calc(x[1], x[0], res[1].sum(), res[0].sum()), axis=1)
        res['iv'] = iv_calc(res[1], res[0])
        res = res.reset_index(drop=False)
        res.columns = ['ref_value', 'count_0', 'count_1', 'ratio_0', 'ratio_1', 'total', 'proportion', 'bad_rate',
                       'new_ref_value', 'iv']

        return res

    def calc_psi(self, datasets):
        """
        计算变量的PSI，按WOE时的分bin区间分bin
        Parameters
        ----------
        datasets: list
            经过Woe替换后的dataframe列表
        """
        if len(datasets) != len(self.datanames):
            raise ValueError('length of datasets does not match length of datanames')
        df_psi = pd.DataFrame(columns=['dataset', 'var_name', 'psi'])

        n = 0
        for i, part in enumerate(self.datanames):
            df_part = datasets[i]
            for var in self.variables['numerical'] + self.variables['categorical']:
                if var in self.variables['numerical']:
                    woe_var = 'nwoe_' + var
                else:
                    woe_var = 'cwoe_' + var
                # 根据WOE reference table计算训练集上的取值占比
                train_prop = self.woe_ref.loc[self.woe_ref['Var_Name'] == var, ['Ref_Value', 'Proportion']]
                # 计算新的数据的取值占比
                delp_value_prop = df_part[woe_var].value_counts() / df_part.shape[0]
                delp_prop = pd.DataFrame(delp_value_prop).reset_index(drop=False)
                delp_prop.columns = ['Ref_Value', 'Delp_Proportion']
                df_prop = pd.merge(train_prop, delp_prop, how='left', on='Ref_Value').dropna()
                psi = calc_psi(df_prop['Proportion'].values, df_prop['Delp_Proportion'].values)
                df_psi.loc[n, 'dataset'] = part
                df_psi.loc[n, 'var_name'] = var
                df_psi.loc[n, 'psi'] = psi
                n += 1

        self.psi_tend = df_psi

    def calc_woe_shift(self, datasets):
        """
        计算变量在不同数据集中各个bin的woe值, 并且根据woe在不用数据集中的偏移情况就算调整后的IV
        Parameters
        ----------
        datasets: list
            经过Woe替换后的dataframe列表
        """
        if len(datasets) != len(self.datanames):
            raise ValueError('length of datasets does not match length of datanames')

        df_woe_tend = pd.DataFrame(columns=['dataset', 'var_name', 'bin_no', 'var_value', 'ref_value', 'new_ref_value',
                                            'count_0', 'count_1', 'total', 'bad_rate', 'proportion', 'iv_adj'])
        for var in self.variables['numerical'] + self.variables['categorical']:
            if var in self.variables['numerical']:
                woe_var = 'nwoe_' + var
            else:
                woe_var = 'cwoe_' + var
            left_table = self.woe_ref.loc[
                self.woe_ref['Var_Name'] == var, ['Var_Name', 'Bin_No', 'Var_Value', 'Ref_Value', 'IV']].reset_index(
                drop=True)
            left_table.columns = ['var_name', 'bin_no', 'var_value', 'ref_value', 'iv']
            left_table['ref_value'] = left_table['ref_value'].apply(lambda x: round(x, 5))

            var_res = []
            for i, part in enumerate(self.datanames):
                df_part = datasets[i]
                # 重新计算每个bin的woe值
                groups = df_part.groupby(woe_var)
                ds = groups[self.label].value_counts().unstack().sort_index().fillna(0)
                ds['ratio_0'] = ds[0] / ds[0].sum()
                ds['ratio_1'] = ds[1] / ds[1].sum()
                ds['total'] = ds[0] + ds[1]
                ds['proportion'] = ds['total'] / df_part.shape[0]
                ds['bad_rate'] = ds[1] / ds['total']
                ds['new_ref_value'] = ds.apply(lambda x: woe_calc(x[1], x[0], ds[1].sum(), ds[0].sum()), axis=1)
                ds['new_iv'] = iv_calc(ds[1], ds[0])
                ds = ds.reset_index(drop=False)
                ds.columns = ['ref_value', 'count_0', 'count_1', 'ratio_0', 'ratio_1', 'total', 'proportion',
                              'bad_rate', 'new_ref_value', 'new_iv']

                # 计算IV的惩罚
                ds['penalty'] = np.abs(ds['ratio_1'] - ds['ratio_0']) * np.abs(ds['new_ref_value'] - ds['ref_value'])
                ds['ref_value'] = ds['ref_value'].apply(lambda x: round(x, 5))  # 取woe值5位小数,防止精度问题merge不上
                ds = pd.merge(left_table, ds, how='left', on=['ref_value'])
                ds['dataset'] = part
                var_res.append(ds)

            woe_tend_var = pd.concat(var_res, axis=0, ignore_index=True)
            woe_tend_var['iv_adj'] = woe_tend_var['new_iv'].mean() - woe_tend_var['penalty'].sum() / len(
                self.datanames)
            df_woe_tend = pd.concat([df_woe_tend, woe_tend_var], axis=0, ignore_index=True)

        df_woe_tend = df_woe_tend[['dataset', 'var_name', 'bin_no', 'var_value', 'total', 'proportion', 'bad_rate',
                                   'ref_value', 'new_ref_value', 'iv', 'new_iv', 'iv_adj']]
        self.woe_tend = df_woe_tend

    def plot_stability_graph(self, to_show=True, save_path=None):
        """
        绘制变量稳定性图
        Parameters
        ----------
        to_show: bool, default True
            是否展示图片
        save_path: str, default None
            图片存储路径
        """
        if self.psi_tend is None or self.woe_tend is None:
            raise ValueError('No monitor result')

        colors = plt.cm.Paired(np.linspace(0.1, 1, len(self.datanames)))
        for var in self.variables['numerical'] + self.variables['categorical']:
            print(var)
            df_psi_var = self.psi_tend.loc[self.psi_tend['var_name'] == var, :].reset_index(drop=True)
            df_woe_tend_var = self.woe_tend.loc[self.woe_tend['var_name'] == var, :].reset_index(drop=True)
            df_proportion_pivot = df_woe_tend_var.pivot_table(index='dataset', columns='bin_no', values='proportion')
            df_woe_value_pivot = df_woe_tend_var.pivot_table(index='dataset', columns='bin_no', values='new_ref_value')
            bins = df_woe_tend_var['bin_no'].unique()

            plt.subplots(1, 2, figsize=(10, 4), dpi=120)
            # PSI图
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(len(self.datanames)), df_psi_var['psi'], color='mediumpurple')
            plt.xticks(np.arange(len(self.datanames)), self.datanames, fontsize=8)
            plt.ylabel('PSI', fontsize=8)
            plt.ylabel('Dataset', fontsize=8)
            plt.title('Average PSI = {0}'.format(round(df_psi_var['psi'].mean(), 4)), fontsize=12)

            # WOE趋势图
            plt.subplot(1, 2, 2)
            for i, part in enumerate(self.datanames):
                plt.bar(x=bins + 0.1 * (i - len(self.datanames) + 1), height=df_proportion_pivot.loc[part, :],
                        width=0.1, color=colors[i], label=part)
            plt.axis(ymin=0, ymax=1.0)
            plt.xticks(bins, bins, fontsize=8)
            plt.ylabel('Proportion', fontsize=8)
            plt.xlabel('Bin number', fontsize=8)
            plt.legend(loc=2, fontsize=6)
            plt.twinx()
            for i, part in enumerate(self.datanames):
                plt.plot(bins, df_woe_value_pivot.loc[part, :], color=colors[i], label=part)
            plt.ylabel('Woe value', fontsize=8)
            # plt.legend(loc=1, fontsize=6)
            plt.title('Adjust IV = {0}'.format(round(df_woe_tend_var['iv_adj'].iloc[0], 5)), fontsize=12)
            plt.suptitle(var, fontsize=16, x=0.5, y=1.04, horizontalalignment='center')
            plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

            if save_path is not None:
                if save_path.endswith('.png') or save_path.endswith('.jpg'):
                    plt.savefig(save_path, bbox_inches='tight')
                elif os.path.isdir(save_path):
                    plt.savefig(os.path.join(save_path, '{0}.png'.format(var)), bbox_inches='tight')
                else:
                    raise ValueError('No such file or directory: {0}'.format(save_path))
            if to_show:
                plt.show()
            plt.close()