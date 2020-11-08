# coding: utf-8
# Author: Jingcheng Qiu

"""
模型效果验证工具包
"""


import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def cut_bins(x, bins=10, method='equal'):
    """
    对x进行分bin，返回每个样本的bin值
    Parameters
    ----------
    x: numpy.ndarray or pandas.Series
        变量
    bins: int or list, default 10
        分bin参数
    method: str, default 'equal', options ['equal', 'quantile', 'point']
        分bin方式，'equal'是等样本量分bin，'quantile'使用分位点分bin, 'threshold'使用指定阈值分bin

    Returns
    -------
    bin_no: numpy.ndarray
        样本的bin值
    """
    if method not in ('equal', 'quantile', 'point'):
        raise ValueError('method only choose "quantile" or "point"')

    if method == 'equal':
        if type(bins) != int:
            raise ValueError('when choose "equal" method, bins need int number')
        bin_no = pd.qcut(x, q=bins, labels=range(1, bins + 1), precision=10).astype(int)
    elif method == 'quantile':
        if type(bins) not in (list, np.ndarray):
            raise ValueError('when choose "quantile" method, bins need list or np.ndarray')
        bin_no = pd.qcut(x, q=bins, labels=range(1, len(bins) + 1), precision=10).astype(int)
    elif method == 'threshold':
        if type(bins) not in (list, np.ndarray):
            raise ValueError('when choose "threshold" method, bins need list or np.ndarray')
        bin_no = np.digitize(x, bins=bins, right=False)

    return bin_no


def eval_model(df, label, score, bins=10, title=None, to_show=True, save_path=None):
    """
    计算模型的TPR、FPR，绘制ROC曲线、TPR-FPR曲线和Sloping曲线
    Parameters
    ----------
    df: pandas.DataFrame
        需包含标签和模型分
    label: numpy.ndarray
        样本的标签
    score: numpy.ndarray
        样本的预测分数
    bins: int, default 10
        分bin个数
    title: str, default None
        图片名称，通常以数据集命名，eg. ins、oos、oot
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存储路径

    Returns
    -------
    df_sloping: DataFrame
        每个bin的target rate和模型分均值
    auc: float
        模型AUC值
    ks: float
        模型ks值
    """
    n = df.shape[0]  # 样本量
    fpr, tpr, thresholds = roc_curve(df[label], df[score])
    diff = tpr - fpr
    auc_value = auc(fpr, tpr)
    ks = diff.max()
    maxidx = 1.0 * diff.argmax() / diff.size
    cut_point = thresholds[diff.argmax()]
    reject_porp = round(100.0 * (predict >= cut_point).sum() / n, 2)

    df_tmp = pd.DataFrame({'truth': truth, 'predict': predict})
    df_tmp['bin'] = cut_bins(df_tmp['predict'], bins=bins, method='equal')
    group = df_tmp.groupby('bin')
    df_sloping = group['truth'].count().to_frame().reset_index(drop=False)
    df_sloping.columns = ['bin', 'sample_count']
    df_sloping['target_rate'] = group['truth'].mean().values
    df_sloping['avg_score'] = group['predict'].mean().values

    # ROC
    plt.figure(figsize=(12, 3), dpi=200)
    plt.subplot(1, 4, 1)
    plt.plot(fpr, tpr, linewidth=0.8)
    plt.plot((0, 1), (0, 1), color='k', linestyle='dashed', linewidth=0.5)
    plt.plot(fpr[diff.argmax()], tpr[diff.argmax()], 'r.', markersize=5)
    plt.axis(xmin=0.0, xmax=1.0)
    plt.axis(ymin=0.0, ymax=1.0)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('false positive rate', fontsize=6)
    plt.ylabel('true positive rate', fontsize=6)
    plt.title('AUC = {0}'.format(round(auc_value, 3)), fontsize=7)

    # score distribution
    plt.subplot(1, 4, 2)
    plt.hist(predict, bins=30, normed=True, facecolor='mediumaquamarine', alpha=0.9)
    plt.axvline(x=np.mean(df_tmp['predict']), color='powderblue', linestyle='dashed', linewidth=0.7)
    plt.axvline(x=np.mean(df_tmp['truth']), color='lightcoral', linestyle='dashed', linewidth=0.7)
    plt.title('Tru = {0},  Pred = {1}'.format(round(df_tmp['truth'].mean(), 3), round(df_tmp['predict'].mean(), 3)), fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('score', fontsize=6)
    plt.ylabel('probability', fontsize=6)

    # TPR-FPR Curve
    plt.subplot(1, 4, 3)
    plt.plot(np.linspace(0, 1, diff.size), tpr, linewidth=0.8, color='cornflowerblue', label='TPR')
    plt.plot(np.linspace(0, 1, diff.size), fpr, linewidth=0.8, color='firebrick', label='FPR')
    plt.plot(np.linspace(0, 1, diff.size), diff, linewidth=0.8, color='slategray', label='TPR - FPR')
    plt.plot((maxidx, maxidx), (0.0, ks), linewidth=0.4, color='r')
    plt.axis(xmin=0.0, xmax=1.0)
    plt.axis(ymin=0.0, ymax=1.0)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.ylabel('tpr / fpr', fontsize=6)
    plt.legend(loc=2, fontsize=6)
    plt.title('KS = {0}, Thres = {1}, Reject {2}%'.format(round(ks, 3), round(cut_point, 4), reject_porp), fontsize=7)

    # Sloping
    plt.subplot(1, 4, 4)
    plt.plot(df_sloping['bin'], df_sloping['avg_score'], 'b.-', linewidth=0.8, label='Prediction', markersize=3)
    plt.plot(df_sloping['bin'], df_sloping['target_rate'], 'r.-', linewidth=0.8, label='Truth', markersize=3)
    plt.axhline(df_tmp['predict'].mean(), color='powderblue', linestyle='dashed', linewidth=0.7, label='Overall Avg score')
    plt.axhline(df_tmp['truth'].mean(), color='lightcoral', linestyle='dashed', linewidth=0.7, label='Overall Target rate')
    plt.legend(loc=2, fontsize=6)
    plt.xticks(df_sloping['bin'], fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('bin', fontsize=6)
    plt.ylabel('target rate', fontsize=6)
    plt.title('Sample = {0}, Bins = {1}'.format(n, df_sloping.shape[0]), fontsize=7)
    if title is not None:
        plt.suptitle(title, fontsize=10, x=0.02, y=1.04, horizontalalignment='left')
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)

    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'model_performance({0}).png'.format(title)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()

    return df_sloping, auc_value, ks


def eval_model_by_period(result, label, score, date, period='month'):
    """
    计算在不同时间窗样本上模型的AUC和KS
    Parameters
    ----------
    result: pandas.DataFrame
        至少包含模型分score, 样本标签label和日期date列
    label: str
        样本标签列名
    score: str
        模型分列名
    date: str
        日期列名
    period: str, default 'month', options ['day', 'week', 'month']
        划分周期

    Returns
    -------
    model_perfms: pandas.DataFrame
        不同时期样本上模型的AUC和KS
    """
    try:
        datetime.strptime(result[date].iloc[0], '%Y-%m-%d')
    except:
        raise Exception('"{0}" is not a correct date format'.format(date))
    if result[score].isnull().sum() > 0:
        print('score contains NaN')
        df_tmp = result.loc[result[score].notnull(), [label, score, date]].copy().reset_index(drop=True)
    else:
        df_tmp = result[[label, score, date]].copy()

    if period == 'day':
        df_tmp['period'] = df_tmp[date].apply(lambda x: x[0:10])
    elif period == 'week':
        df_tmp['period'] = '9999-12-31'
        week_start_date = df_tmp[date].min()
        week_end_date = week_start_date
        max_date = df_tmp[date].max()
        while week_end_date < max_date:
            week_end_date = datetime.strftime(datetime.strptime(week_start_date, '%Y-%m-%d') + timedelta(7), '%Y-%m-%d')
            df_tmp.loc[(df_tmp[date] >= week_start_date) & (df_tmp[date] < week_end_date), 'period'] = week_start_date
            week_start_date = week_end_date
    elif period == 'month':
        df_tmp['period'] = df_tmp[date].apply(lambda x: x[0:7])
    else:
        raise ValueError('period can only be "day", "week" or "month"')

    period_list = np.sort(df_tmp['period'].unique())
    sample_cnt = []
    auc_by_period = []
    ks_by_period = []
    for p in period_list:
        part = df_tmp.loc[df_tmp['period'] == p, [label, score]]
        fpr, tpr, thresholds = roc_curve(part[label].values, part[score].values)
        diff = tpr - fpr
        sample_cnt.append(part.shape[0])
        auc_by_period.append(auc(fpr, tpr))
        ks_by_period.append(diff.max())
    model_perfms = pd.DataFrame()
    model_perfms['period'] = period_list
    model_perfms['sample_cnt'] = sample_cnt
    model_perfms['AUC'] = auc_by_period
    model_perfms['KS'] = ks_by_period

    return model_perfms


def eval_model_by_group(result, label, score, group):
    """
    计算不同群体样本上模型的AUC和KS指标
    Parameters
    ----------
    result: pandas.DataFrame
        至少包含模型分score, 样本标签label和分群标签group列
    label: str
        样本标签列名
    score: str
        模型分列名
    group: str
        分群列名

    Returns
    -------
    model_perfms: pandas.DataFrame
        不同时期样本上模型的AUC和KS
    """
    if result[score].isnull().sum() > 0:
        print('score contains NaN')
        df_tmp = result.loc[result[score].notnull(), [label, score, group]].copy().reset_index(drop=True)
    else:
        df_tmp = result[[label, score, group]].copy()

    group_list = np.sort(df_tmp[group].unique())
    sample_cnt = []
    auc_by_period = []
    ks_by_period = []
    for g in group_list:
        part = df_tmp.loc[df_tmp[group] == g, [label, score]]
        fpr, tpr, thresholds = roc_curve(part[label].values, part[score].values)
        diff = tpr - fpr
        sample_cnt.append(part.shape[0])
        auc_by_period.append(auc(fpr, tpr))
        ks_by_period.append(diff.max())
    model_perfms = pd.DataFrame()
    model_perfms['group'] = group_list
    model_perfms['sample_cnt'] = sample_cnt
    model_perfms['AUC'] = auc_by_period
    model_perfms['KS'] = ks_by_period

    return model_perfms


def calc_dp_by_bin(df, score, target, principal=None, dueamount=None, bins=10):
    """
    根据score分bin，计算每个bin的逾期率和累计逾期率
    Parameters
    ----------
    df: pandas.DataFrame
    score: str
        分数列名
    target: str
        target列名
    principal: str, default None
        借款本金
    dueamount: str, default None
        逾期金额
    bins: int or list, default 10
        分bin的方式，如果是int则表示将模型分从小到大排列后均匀分成几个bin，如果是list则按指定切分点分bin

    Returns
    -------
    result: pandas.DataFrame
        每个bin中的逾期率
    """
    df_result = pd.DataFrame()
    df_tmp = df[[score, target]].copy()
    if type(bins) == int:
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='equal')
    elif type(bins) in (list, np.ndarray):
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='quantile')
    else:
        raise ValueError('bins type can only be [int, list, np.ndarray]')

    if principal is not None and dueamount is not None:
        df_tmp[principal] = df[principal]
        df_tmp[dueamount] = df[dueamount]
    group = df_tmp.groupby('bin')
    sample_cnt = group[target].count()
    overdue_cnt = group[target].sum()
    df_result['bin'] = np.arange(1, len(sample_cnt) + 1)
    df_result['sample'] = sample_cnt.values
    df_result['dp'] = (overdue_cnt / sample_cnt).values
    df_result['cum_dp'] = (overdue_cnt.cumsum() / sample_cnt.cumsum()).values
    if principal is not None and dueamount is not None:
        total_principal = group[principal].sum()
        total_dueamount = group[dueamount].sum()
        df_result['prin_ratio'] = (total_principal / total_principal.sum()).values
        df_result['amt_dp'] = (total_dueamount / total_principal).values
        df_result['cum_amt_dp'] = (total_dueamount.cumsum() / total_principal.cumsum()).values

    return df_result


def calc_cross_dp(df, score_main, score_ext, split_bin_num=(5, 5), target=None, principal=None, dueamount=None):
    """
    Cross两个模型分计算逾期率
    Parameters
    ----------
    df: pandas.DataFrame
    score_main: str
        主模型分
    score_ext: str
        待检验的模型分
    split_bin_num: list or tuple, defalut(5, 5)
        主模型分和待检验模型分各自的分箱数
    target: str, default None
        样本逾期标签
    principal: str, default None
        借款本金
    dueamount: str, default None
        逾期金额

    Returns
    -------
    distribution: pandas.DataFrame
        两个模型分cross后的bin值分布
    ovd_rate: pandas.DataFrame
        cross后的逾期率
    prin_ratio: pandas.DataFrame
        cross后每个bin的成交金额占比
    amt_ovd_rate: pandas.DataFrame
        cross后的金额逾期率
    """
    bin_main = cut_bins(df[score_main], bins=split_bin_num[0], method='equal')
    bin_ext = cut_bins(df[score_ext], bins=split_bin_num[1], method='equal')

    distribution = pd.crosstab(index=bin_main, columns=bin_ext, normalize=True, margins=True)
    ovd_rate = None
    prin_ratio = None
    amt_ovd_rate = None

    if target is not None:
        ovd_rate = pd.crosstab(index=bin_main, columns=bin_ext, values=df[target], aggfunc='mean', margins=True)

    if principal is not None and dueamount is not None:
        prin_ratio = pd.crosstab(index=bin_main, columns=bin_ext, values=df[principal], aggfunc='sum', margins=True) / df[principal].sum()
        amt_ovd_rate = pd.crosstab(index=bin_main, columns=bin_ext, values=df[dueamount], aggfunc='sum', margins=True) / \
                       pd.crosstab(index=bin_main, columns=bin_ext, values=df[principal], aggfunc='sum', margins=True)

    return distribution, ovd_rate, prin_ratio, amt_ovd_rate


def bad_sample_rank_scatter(df, score, target, cutoff=None, title=None, save_path=None):
    """
    绘制坏样本模型分排序的散点图
    df: DataFrame
    score: str
        分数的列名
    target: str
        target列名
    cutoff: list, default None
        切分点，切分的百分比, eg. [0.05, 0.2, 0.5, 0.8, 0.9]
    title: str, default None
        图片标题
    save_path: str, default None
        图片存储路径
    """
    score_rank = df[score].rank(method='first').astype(int)  # 模型分排序
    # good_sample_rank = score_rank.loc[df[target] == 0].values
    bad_sample_rank = score_rank.loc[df[target] == 1].values

    plt.figure(figsize=(16, 5), dpi=400)
    #     plt.scatter(x=good_sample_rank, y=np.repeat(1, good_sample_rank.shape[0]),
    #                 s=np.repeat(0.1, good_sample_rank.shape[0]), c='w', label='good')
    plt.scatter(x=bad_sample_rank, y=np.repeat(1, bad_sample_rank.shape[0]),
                s=np.repeat(0.4, bad_sample_rank.shape[0]), c='r', label='bad sample')

    # 加垂直分割线
    if cutoff is not None:
        cut_rank = score_rank.quantile(cutoff)
        for p in cut_rank:
            plt.axvline(p, color='gray', linestyle='dashed', linewidth=0.6)
    plt.legend(loc=1, fontsize=12)
    plt.axis(xmin=0.0, xmax=score_rank.max())
    plt.axis(ymin=0.0, ymax=2)
    plt.yticks([], fontsize=8)
    plt.xlabel('rank', fontsize=10)
    if title is not None:
        plt.title(title, fontsize=15)

    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'bad_sample_scatter.png'), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    plt.show()
    plt.close()


def meanofbucket(df, x, score, y, type='numerical', bins=10, cut_method='equal', to_show=True, save_path=None):
    """
    根据变量x分bin，计算每个bin中样本的score的均值和target rate
    Parameters
    ----------
    df: pandas.DataFrame
    x: str
        用于分bin的变量名
    score: str
        score列名
    y: str
        y列名
    type: str, default 'numerical' options ['numerical', 'categorical']
        变量类型
    bins: int or list, default 10
        分bin参数
    cut_method: str, default 'equal', options ['equal', 'quantile', 'point']
        分bin方式，'equal'是等样本量分bin，'quantile'使用分位点分bin, 'threshold'使用指定阈值分bin
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存储路径

    Returns
    -------
    result: pandas.DataFrame
        每个bin中变量的均值
    """
    df_tmp = df[[x, score, y]].copy()
    if type == 'numerical':
        df_tmp['bin'] = cut_bins(df_tmp[x], bins=bins, method=cut_method)
    elif type == 'categorical':
        df_tmp['bin'] = df_tmp[x]
    else:
        raise ValueError('type can only be "numerical" or "categorical"]')

    group = df_tmp.groupby('bin')
    result = group[x].count().to_frame().reset_index(drop=False)
    result.columns = ['bin', 'sample_count']
    # result['bin_lb'] = cut_points
    result['proportion'] = 1.0 * result['sample_count'] / df_tmp.shape[0]
    # result['bin'] = result['bin'].astype(int)
    result['prediction'] = group[score].mean().values
    result['truth'] = group[y].mean().values
    result['residual'] = np.abs(result['truth'] - result['prediction'])

    x_coor = np.arange(result.shape[0])
#    if type == 'numerical':
#        shift = np.arange(result.shape[0]) + 0.5
#        xticks = result['bin']
#    elif type == 'categorical':
#        shift = np.arange(result.shape[0])
#        xticks = result['bin']
    xticks = result['bin']
    
    plt.figure(figsize=(8, 3), dpi=200)
    plt.subplot(1, 2, 1)
    plt.plot(x_coor, result['truth'], color='firebrick', linewidth=0.7, marker='.', markersize=2,
             label='truth')
    plt.plot(x_coor, result['prediction'], color='cornflowerblue', linewidth=0.7, marker='.', markersize=2,
             label='prediction')
    plt.plot(x_coor, result['residual'], color='mediumaquamarine', linestyle='dashed', linewidth=0.7,
             label='residual')
    plt.axis(ymin=0.0, ymax=result[['prediction', 'truth']].max().max() * 1.5)
    plt.xticks(x_coor, xticks, rotation=60, fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('bin', fontsize=6)
    plt.ylabel('mean', fontsize=6)
    plt.legend(loc=2, fontsize=6)
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.4)
    plt.title('Mean of prediction and truth in each bucket', fontsize=7)

    plt.subplot(1, 2, 2)
    plt.bar(x_coor, result['proportion'], color='lightgray', align='center', width=0.95)
    plt.xticks(x_coor, xticks, rotation=60, fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('bin', fontsize=6)
    plt.ylabel('proportion', fontsize=6)
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.4)
    plt.title('Sample Proportion in each bucket', fontsize=7)

    plt.suptitle(x, fontsize=10, x=0.5, y=1.01)
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '{0}.png'.format(x)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()

    return result


#
# def swap_in_out_anlys(df_score, score_new, score_old, amount, target, ind_deal, buffer=1.0, bin_num=10):
#     header = ['passrate', 'overdue_rt_diff', 'overdue_rt_diff_rto', 'amount_overdue_rt_diff',
#               'amount_overdue_rt_diff_rto']
#     df_result = pd.DataFrame(columns=header)
#
#     # 计算切分点并分bin
#     cut_points_new = df_score[score_new].quantile(np.arange(0.0, 1.0, 1 / bin_num))
#     cut_points_old = df_score[score_old].quantile(np.arange(0.0, 1.0, 1 / bin_num))
#     cut_points_new = np.append(cut_points_new, 1.0)
#     cut_points_old = np.append(cut_points_old, 1.0)
#     df_score['bin_new'] = pd.cut(df_score[score_new], bins=cut_points_new, precision=13, right=True,
#                                  include_lowest=True, labels=range(1, bin_num + 1))
#     df_score['bin_old'] = pd.cut(df_score[score_old], bins=cut_points_old, precision=13, right=True,
#                                  include_lowest=True, labels=range(1, bin_num + 1))
#     df_score_deal = df_score[df_score[ind_deal] == 1].reset_index(drop=True)
#     df_score_overdue = df_score[df_score[target] == 1].reset_index(drop=True)
#
#     # 计算交叉矩阵
#     cnt_mat = pd.crosstab(df_score['bin_new'], df_score['bin_old']).fillna(0)  # 新老模型每个bin的人数的交叉矩阵
#     overdue_rt_mat = df_score_deal.pivot_table(index='bin_new', columns='bin_old', values=target,
#                                                aggfunc='mean', fill_value=0.0)  # 新老模型每个bin的逾期率
#     amount_mat = df_score.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                       aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的发标金额
#     deal_amount_mat = df_score_deal.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                                 aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的发标金额
#     overdue_amount_mat = df_score_overdue.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                                       aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的逾期金额
#     amount_overdue_rt_mat = (overdue_amount_mat / deal_amount_mat).fillna(0.0)
#
#     for i, passrate in enumerate(np.arange(0.0 + 1.0 / bin_num, 1.0, 1.0 / bin_num)):
#         overdue_rt_new = (cnt_mat.iloc[0:i + 1, :] * overdue_rt_mat.iloc[0:i + 1, :]).sum().sum() / cnt_mat.iloc[
#                                                                                                     0:i + 1,
#                                                                                                     :].sum().sum()
#         overdue_rt_old = (cnt_mat.iloc[:, 0:i + 1] * overdue_rt_mat.iloc[:, 0:i + 1]).sum().sum() / cnt_mat.iloc[:,
#                                                                                                     0:i + 1].sum().sum()
#         amount_overdue_rt_new = (amount_mat.iloc[0:i + 1, :] * amount_overdue_rt_mat.iloc[0:i + 1,
#                                                                :]).sum().sum() / amount_mat.iloc[0:i + 1, :].sum().sum()
#         amount_overdue_rt_old = (amount_mat.iloc[:, 0:i + 1] * amount_overdue_rt_mat.iloc[:,
#                                                                0:i + 1]).sum().sum() / amount_mat.iloc[:,
#                                                                                        0:i + 1].sum().sum()
#
#         df_result.loc[i, 'passrate'] = passrate
#         df_result.loc[i, 'overdue_rt_diff'] = overdue_rt_old - overdue_rt_new
#         df_result.loc[i, 'overdue_rt_diff_rto'] = (overdue_rt_old - overdue_rt_new) / overdue_rt_old
#         df_result.loc[i, 'amount_overdue_rt_diff'] = amount_overdue_rt_old - amount_overdue_rt_new
#         df_result.loc[i, 'amount_overdue_rt_diff_rto'] = (
#                                                          amount_overdue_rt_old - amount_overdue_rt_new) / amount_overdue_rt_old
#
#     return df_result
#
#
# def shift_matrix(df_base, df_shift, var_name, join_key):
#     """
#     计算转移矩阵
#     """
#     if type(join_key) == str:
#         var_list = [join_key] + [var_name]
#     elif type(join_key) == list:
#         var_list = join_key + [var_name]
#
#     df_merge = pd.merge(df_base[var_list], df_shift[var_list], how='inner', on=join_key, suffixes=['_base', '_shift'])
#
#     if df_merge.shape[0] != df_base.shape[0]:
#         print('base数据与shift数据的scope不完全重合')
#     mat = pd.crosstab(df_merge[var_name + '_base'], df_merge[var_name + '_shift'], margins=True, normalize='index')
#
#     return mat
