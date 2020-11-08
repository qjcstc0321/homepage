# coding: utf-8
# Author: Jingcheng Qiu

"""
建模工具包
"""


import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def slice_csv_file(file_path, save_path, nrows_per=-1, ncols_per=-1):
    """
    将csv文件按行和列切分成多个文件，方便读取
    Parameters
    ----------
    file_path: str
        待拆分的csv文件路径
    save_path: str
        拆分后的csv文件存放路径
    nrows_per: int, default -1
        拆分后每份文件包含的行数, -1表示不按行拆分
    ncols_per: int, default -1
        拆分后每份文件包含的列数, -1表示不按列拆分
    """
    # 获取csv文件的总行数、列数和列名
    nrow = -1    # 初始值设为-1是减去表头那一行
    with open(file_path, mode='rb') as f:
        for _ in f:
            nrow += 1
    tmp = pd.read_csv(file_path, nrows=1, header=0)    # 需确保csv文件有表头
    ncol = tmp.shape[1]     # 总列数
    colnames = tmp.columns.tolist()    # 表头
    file_name = os.path.basename(file_path)

    if nrows_per == -1 and nrows_per == -1:
        raise ValueError('nrows_per and nrows_per can not both equal -1')
    if nrows_per == -1:
        nrows_per = nrow
    if ncols_per == -1:
        ncols_per = ncol

    file_count = 0
    startrow = 1
    while True:
        if startrow > nrow:
            break
        if startrow + nrows_per > nrow:
            endrow = nrow
        else:
            endrow = startrow + nrows_per - 1
        df_partrow = pd.read_csv(file_path, nrows=nrows_per, skiprows=startrow, header=None, names=colnames)

        startcol = 1
        while True:
            if startcol > ncol:
                break
            if startcol + ncols_per > ncol:
                endcol = ncol
            else:
                endcol = startcol + ncols_per - 1
            df_partcol = df_partrow.iloc[:, (startcol-1):endcol]
            df_partcol.to_csv(os.path.join(save_path, '({0}-{1}行{2}-{3}列){4}'.format(startrow, endrow, startcol, endcol, file_name)), index=False)
            startcol += ncols_per
            file_count += 1
            print('\r', '{} files save successfully'.format(file_count), end='', flush=True)
        startrow += nrows_per

    return True


def down_sample(df, label, fraction=0.5, random_state=100):
    """
    对负样本进行降采样
    Parameters
    ----------
    df: pandas.DataFrame
    label: str
        label列名
    fraction: float, default 0.5
        正样本占抽样后数据的比例, 取值范围为[0, 1]之间
    random_state: int, default 100
        抽样的随机数种子

    Returns
    -------
    df_sample: pandas.DataFrame
        抽样后的数据集
    """
    if fraction >= 1.0 or fraction <= 0.0:
        raise ValueError('"fraction" must be between 0 and 1')
    df_positive = df.loc[df[label] == 1, label]
    df_negative = df.loc[df[label] == 0, label]
    negative_samples = int(df_positive.size * (1 - fraction) / fraction)    # 计算当前正样本占比下负样本的数量
    if negative_samples > df_negative.shape[0]:
        raise ValueError('The negative sample size does not support {} sampling ratio'.format(fraction))

    df_negative_sample = train_test_split(df_negative, test_size=negative_samples, random_state=random_state)[1]
    sample_index = pd.concat([df_positive, df_negative_sample]).sort_index().index
    df_sample = df.loc[sample_index, :].reset_index(drop=True)
    print('降采样后样本数: {0}'.format(df_sample.shape[0]))

    return df_sample


def split_data_set(df, label, train_size=0.7, random_state=100):
    """
    将DataFrame划分成训练集和测试集
    Parameters
    ----------
    df: pandas.DataFrame
    label: str
        label列名
    train_size: float, default 0.7
        训练集原本占比
    random_state: int, default 100
        随机数种子

    Returns
    -------
    df_train: pandas.DataFrame
        训练数据集
    df_test: pandas.DataFrame
        测试数据集
    """
    if train_size >= 1.0 or train_size <= 0.0:
        raise ValueError('"train_size" must be between 0 and 1')
    df_train, df_test, y_train, y_test = train_test_split(df, df[label], test_size=1-train_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print('训练样本数: {0}, 测试样本数: {1}'.format(df_train.shape[0], df_test.shape[0]))

    return df_train, df_test


def calc_vif(df, save_path=None):
    """
    计算每个变量的方差膨胀因子
    Parameters
    ----------
    df: pandas.DataFrame
    save_path: str, default None
        结果存放路径

    Returns
    -------
    df_vif: pandas.DataFrame
    """
    df_vif = pd.DataFrame(columns=['var_name', 'vif'])
    df_vif['var_name'] = df.columns
    df_vif['vif'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    df_vif = df_vif.sort_values(by='vif', ascending=False)

    if save_path is not None:
        df_vif.to_csv(save_path, index=False)
    
    return df_vif


def repetitive_check(df):
    """
    检查DataFrame中是否有完全相同的行或者列
    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    same_row：DataFrame
    same_col：list
        相同的列，成对展示
    """
    same_col = []
    col_names = df.columns

    # check repetitive columns
    for i in range(len(col_names)):
        for j in range(i+1, len(col_names)):
            col_i = col_names[i]
            col_j = col_names[j]
            mean_i = df[col_i].mean()
            mean_j = df[col_j].mean()
            std_i = df[col_i].std()
            std_j = df[col_j].std()
            if abs(mean_i - mean_j) < 1e-6 and abs(std_i - std_j) < 1e-6:
                same_col.append([col_i, col_j])

    # check repetitive rows
    same_row = df[df.duplicated(keep=False)]
    
    return same_row, same_col


def find_high_corr_var_pair(df, corr_thres=0.8):
    """
    查找DataFrame中相关性较高的变量组
    Parameters
    ----------
    df: DataFrame
    corr_thres: 相关系数阈值，相关性大于该阈值的变量对会被输出

    Returns
    -------
    corr_mat：DataFrame
        所有变量相关系数矩阵的DataFrame
    df_high_corr_pair：DataFrame
        相关系数较大的变量组
    """
    corr_mat = df.corr()
    col_names = corr_mat.columns
    high_corr_pair = []
    for i in range(len(col_names)):
        for j in range(i+1, len(col_names)):
            if abs(corr_mat.iloc[i, j]) > corr_thres:
                high_corr_pair.append([col_names[i], col_names[j], abs(corr_mat.iloc[i, j])])
    df_high_corr_pair = pd.DataFrame(high_corr_pair, columns=['var1', 'var2', 'corr'])
    df_high_corr_pair = df_high_corr_pair.sort_values(by='corr', ascending=False)
    
    return corr_mat, df_high_corr_pair


def train_lr(df, input_vars, target, regular='l1', alpha=0.0, maxiter=1000):
    """
    训练logistic regression模型
    Parameters
    ----------
    df: DataFrame
        训练数据集
    input_vars: list
        进入模型的变量
    target: str
        target列名
    regular: str, default 'l1', options ['l1', 'l2']
        惩罚方式
    alpha: float, default 0.0
        惩罚系数
    maxiter: int, default 1000
        最大迭代次数

    Returns
    -------
    model：object
        训练好的lr模型
    """
    if 'const' not in input_vars:
        input_vars = input_vars + ['const']

    x = df[input_vars]
    y = df[target]
    model = sm.Logit(y, x).fit_regularized(method=regular, alpha=alpha, maxiter=maxiter, trim_mode='size')

    print('============ Logistic Regresion Model Summary============\n')
    print('{0}'.format(model.summary()))

    return model


def choose_l1_param(df_train, df_test, input_vars, target, epoch=30, delta=10.0, to_plot=True):
    """
    逐渐增大L1正则化参数的值，计算模型的AUC和KS和剩余变量个数，帮助选择正则化参数
    Parameters
    ----------
    df_train: DataFrame
        训练数据集
    df_test: DataFrame
        测试数据集
    input_vars: list
        进入模型的变量
    target: str
        target列名
    epoch: int, default 30
        训练轮数
    delta: float, default 10.0
        每轮训练惩罚系数的增长值
    to_plot: bool, default True
        是否展示每轮训练的结果图

    Returns
    -------
    df_return: DataFrame
        每轮训练的结果
    """
    df_train_x = df_train[input_vars]
    df_test_x = df_test[input_vars]
    df_train_y = df_train[target]
    df_test_y = df_test[target]
    df_train_x['const'] = 1.0
    df_test_x['const'] = 1.0
    alpha = np.arange(0.0, epoch*delta, delta)
    res = {'alpha': list(alpha), 'train_ks': [], 'test_ks': [], 'train_auc': [], 'test_auc': [], 'DF': []}

    for i, alp in enumerate(alpha):
        print('========================== Epoch {0}/{1} =========================='.format(i+1, epoch))
        model = sm.Logit(df_train_y, df_train_x).fit_regularized(method='l1', alpha=alp, maxiter=1000.0, trim_mode='size')
        train_predict = model.predict(df_train_x)
        test_predict = model.predict(df_test_x)

        train_fpr, train_tpr, train_thresholds = roc_curve(df_train_y, train_predict)
        test_fpr, test_tpr, test_thresholds = roc_curve(df_test_y, test_predict)
        train_ks = round(max(train_tpr - train_fpr), 4)
        test_ks = round(max(test_tpr - test_fpr), 4)
        train_auc = round(auc(train_fpr, train_tpr), 4)
        test_auc = round(auc(test_fpr, test_tpr), 4)
        DF = model.df_model
        res['train_ks'].append(train_ks)
        res['test_ks'].append(test_ks)
        res['train_auc'].append(train_auc)
        res['test_auc'].append(test_auc)
        res['DF'].append(DF)

        print('alpha:{0}, DF:{1}, train_ks:{2}, test_ks:{3}, train_auc:{4}, test_auc:{5}'.format(alp, DF, train_ks, test_ks, train_auc, test_auc))
    
    df_return = pd.DataFrame(res)
    if to_plot:
        print('========================== Show performance ==========================')
        plt.subplots(1, 3, figsize=(15, 5), dpi=200)
        plt.subplot(1, 3, 1)
        plt.plot(df_return['alpha'], df_return['DF'], color='red', linestyle='-')
        plt.xlabel('alpha', fontsize=15)
        plt.ylabel('Degree of Freedom', fontsize=13)
        
        plt.subplot(1, 3, 2)
        plt.plot(df_return['alpha'], df_return['train_ks'], color='salmon', linestyle='-', label='KS of train set')
        plt.plot(df_return['alpha'], df_return['test_ks'], color='c', linestyle='-', label='KS of test set')
        plt.legend(loc=1, fontsize=10)
        plt.xlabel('alpha', fontsize=13)
        plt.ylabel('KS', fontsize=13)
        
        plt.subplot(1, 3, 3)
        plt.plot(df_return['alpha'], df_return['train_auc'], color='salmon', linestyle='-', label='AUC of train set')
        plt.plot(df_return['alpha'], df_return['test_auc'], color='c', linestyle='-', label='AUC of test set')
        plt.legend(loc=1, fontsize=10)
        plt.xlabel('alpha', fontsize=13)
        plt.ylabel('AUC', fontsize=13)
        
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()
    
    return df_return[['alpha', 'DF', 'train_ks', 'test_ks', 'train_auc', 'test_auc']]


def dynamic_learningrate_gbdt(params, train_set, total_boost_rounds, change_per_rounds, early_stopping_rounds,
                              init_learning_rate, change_rate=0.5, change_type='prod', valid_sets=None, valid_names=None):
    """
    逐渐变化学习率训练lightgbm模型
    Parameters
    ----------
    params: dict
        lightgbm模型参数, 注意参数中不要有learning_rate
    train_set: lightgbm.DataSet object
        训练数据集
    total_boost_rounds: int
        总训练轮数
    change_per_rounds: int
        每多少轮改变学习率
    early_stopping_rounds: int
        当度量指标多少轮不增长就提前停止训练
    init_learning_rate: float
        初始学习率
    change_rate: float
        每次学习率的变化比率
    change_type: str, options ["prod", "sub"]
        学习率变化方式, prod是乘上change_rate，sub是减去change_rate
    valid_sets: lightgbm.DataSet object, default None
        验证数据集
    valid_names: str or tuple, default None
        验证数据集名称

    Returns
    -------
    booster: lightgbm.booster object
    """
    if change_per_rounds > total_boost_rounds:
        raise ValueError('"change_per_rounds" must less than "total_boost_rounds"')

    from math import ceil
    epoch = ceil(total_boost_rounds / change_per_rounds)
    learning_rates = []
    for i in range(epoch):
        boost_rounds = min(change_per_rounds, total_boost_rounds - i * change_per_rounds)
        if change_type == 'prod':
            learning_rates.extend([init_learning_rate * change_rate**i] * boost_rounds)
        elif change_type == 'sub':
            learning_rates.extend([init_learning_rate - change_rate * i] * boost_rounds)
        else:
            raise ValueError('change_type can only be "prod" or "sub"')

    if np.min(learning_rates) <= 0:
        raise ValueError('learning_rate must greater than zero')
    tmp = []
    for a in learning_rates:
        if a not in tmp:
            tmp.append(a)
    print('learning_rates: ', tmp)
    booster = lgb.train(params, num_boost_round=total_boost_rounds, learning_rates=learning_rates,
                        train_set=train_set, valid_sets=valid_sets, valid_names=valid_names,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=1)

    return booster


def get_leaf_num(gbm):
    """
    获取lightgbm模型每棵树的叶子节点数量
    Parameters
    ----------
    gbm: lightgbm.Booster object
        训练好的lightbm模型

    Returns
    -------
    num_leaves: list
        每颗树叶子节点的数量
    """
    model_details = gbm.dump_model()
    num_leaves = []
    for tree in model_details['tree_info']:
        num_leaves.append(tree['num_leaves'])

    return num_leaves


def leafid_to_onehot(leafid, gbm):
    """
    将lightgbm模型输出的叶节点id进行One-hot编码
    Parametes
    ---------
    leafid: numpy.ndarray
        每个样本的叶子节点编号, 二维整数数组, 列数应与模型中树的棵树一致
    gbm: lightgbm.Booster object
        训练好的lightgbm模型

    Returns
    -------
    onehot_mat: scipy.sparse.csr_matrix

    Examples
    --------
    >>> leafid = np.asarray([[3, 3, 0],
                             [2, 1, 0],
                             [0, 0, 1]])
    >>> onehot_mat.toarray()
    array([[0., 0., 0., 1., 0., 0., 0., 1., 1., 0.],
           [0., 0., 1., 0., 0., 1., 0., 0., 1., 0.],
           [1., 0., 0., 0., 1., 0., 0., 0., 0., 1.]])
    """
    if leafid.ndim != 2:
        raise ValueError('array dimension must be 2')
    num_leaves = get_leaf_num(gbm)
    if len(num_leaves) != leafid.shape[1]:
        raise ValueError('The number of columns of `leafid` does not match the number of trees')

    stack = []
    for tree_id, num_leaf in enumerate(num_leaves):
        onehot_mat_1tree = int_onehot_encoder(leafid[:, tree_id], num_leaf, to_sparse=True)
        stack.append(onehot_mat_1tree)
        print('\r', 'Transform completed {0}/{1} trees'.format(tree_id + 1, len(num_leaves)), end='', flush=True)
    onehot_mat = sparse.hstack(stack, format='csr')

    return onehot_mat


def __get_feature_importance(gbm, end_iter=None):
    """
    统计模型的特征重要性
    Parameters
    ----------
    gbm: lightgbm.basic.Booster
        训练好的Lightgbm模型
    end_iter: int, default None
        截止迭代次数，若end_iter > 0则只统计到end_iter以前的树

    Returns
    -------
    df_result: pandas.DataFrame
        特征重要性统计结果
    """
    details = gbm.dump_model()    # ravel model to dict
    split_count = np.zeros(len(details['feature_names']))
    split_gains = np.zeros(len(details['feature_names']))
    split_cover = np.zeros(len(details['feature_names']))
    df_result = pd.DataFrame({'var_name': details['feature_names']})

    if end_iter is None or end_iter <= 0:
        end_iter = len(details['tree_info'])

    for i, tree in enumerate(details['tree_info']):
        if i + 1 > end_iter:
            break
        queue = [tree['tree_structure']]    # 将根节点放入队列
        # BFS遍历每棵树的节点
        while queue:
            node = queue.pop(0)
            if 'leaf_value' in node:
                break
            split_varid = node['split_feature']
            split_count[split_varid] += 1
            split_gains[split_varid] += node['split_gain']
            split_cover[split_varid] += node['internal_count']
            if 'leaf_index' not in node['left_child'].keys():
                queue.append(node['left_child'])
            if 'leaf_index' not in node['right_child'].keys():
                queue.append(node['right_child'])
    df_result['split'] = split_count
    df_result['total_gain'] = split_gains
    df_result['cover'] = split_cover / split_count

    return df_result


def show_feature_importance(obj, end_iter=None, top_n=None, to_show=True, save_path=None):
    """
    输出LGBM模型的feature importance，并绘制条形图
    Parameters
    ----------
    obj: lightgbm.Booster object or pandas.DataFrame
        训练好的Lightgbm模型，或是已经计算好的feature importance DataFrame
    end_iter: int, default None
        截止迭代次数，若end_iter > 0则只统计到end_iter以前的树
    top_n: int, default None
        展示重要性最高的N变量，默认展示全部变量，为保证显示效果，建议当变量个数多于30个时进行限制
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存放路径

    Returns
    -------
    df_fi: pandas.DataFrame
        模型的feature importance，包括以下几个指标:
        'split': 被节点选为分裂特征的次数
        'total_gain': 作为分裂特征时对损失函数的总增益
        'cover': 作为分裂特征时平均每次覆盖的样本数量
        'avg_gain': 作为分裂特征时平均每次对损失函数的增益
        'split_weight': 单个特征的分裂次数占总分裂次数的比例
        'gain_weight': 单个特征的增益占总分裂增益的比例
    """
    if obj.__class__.__name__ == 'LGBMClassifier' or obj.__class__.__name__ == 'Booster':
        if obj.__class__.__name__ == 'LGBMClassifier':
            booster = obj.booster_
        else:
            booster = obj
        df_fi = __get_feature_importance(booster, end_iter)
        df_fi['avg_gain'] = df_fi['total_gain'] / df_fi['split']
        df_fi['split_weight'] = df_fi['split'] / df_fi['split'].sum()
        df_fi['gain_weight'] = df_fi['total_gain'] / df_fi['total_gain'].sum()
        df_fi['gain_rank'] = df_fi['total_gain'].rank(method='first', ascending=False).values.reshape((-1,))
        df_fi['split_rank'] = df_fi['split'].rank(method='first', ascending=False).values.reshape((-1,))
        df_fi['avg_gain_rank'] = df_fi['avg_gain'].rank(method='first', ascending=False).values.reshape((-1,))
        df_fi['cover_rank'] = df_fi['cover'].rank(method='first', ascending=False).values.reshape((-1,))

    elif isinstance(obj, pd.DataFrame):
        df_fi = obj
    else:
        raise ValueError('Unknown object type')

    if top_n is not None:
        df_gain_fi = df_fi.loc[df_fi['gain_rank'] <= top_n, :].copy().sort_values(by='gain_rank', ascending=False)
        df_split_fi = df_fi.loc[df_fi['split_rank'] <= top_n, :].copy().sort_values(by='split_rank', ascending=False)
        df_cover_fi = df_fi.loc[df_fi['cover_rank'] <= top_n, :].copy().sort_values(by='cover_rank', ascending=False)
        title1 = 'Weight of Split Gain (Top {0})'.format(top_n)
        title2 = 'Weight of Split Count (Top {0})'.format(top_n)
        title3 = 'Sample Coverage across all splits (Top {0})'.format(top_n)
    else:
        df_gain_fi = df_fi.copy().sort_values(by='gain_rank', ascending=False)
        df_split_fi = df_fi.copy().sort_values(by='split_rank', ascending=False)
        df_cover_fi = df_fi.copy().sort_values(by='cover_rank', ascending=False)
        title1 = 'Weight of Split Gain'
        title2 = 'Weight of Split Count'
        title3 = 'Sample coverage across all splits'

    plt.figure(figsize=(4, 9), dpi=200)
    plt.subplot(3, 1, 1)
    plt.barh(np.arange(df_gain_fi.shape[0]), df_gain_fi['gain_weight'], height=0.6, color='lightskyblue')
    for i, var in enumerate(df_gain_fi['var_name']):
        plt.annotate(var, xy=(0.001, i), va='center', ha='left', fontsize=4, color='black', fontweight='normal')
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.7)
    plt.xticks(fontsize=5)
    plt.yticks([])
    plt.xlabel('gain weight', fontsize=5)
    plt.title(title1, fontsize=6)

    plt.subplot(3, 1, 2)
    plt.barh(np.arange(df_gain_fi.shape[0]), df_split_fi['split_weight'], height=0.6, color='lightgreen')
    for i, var in enumerate(df_split_fi['var_name']):
        plt.annotate(var, xy=(0.001, i), va='center', ha='left', fontsize=4, color='black', fontweight='normal')
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.7)
    plt.xticks(fontsize=5)
    plt.yticks([])
    plt.xlabel('split weight', fontsize=5)
    plt.title(title2, fontsize=6)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=2.0)

    plt.subplot(3, 1, 3)
    plt.barh(np.arange(df_gain_fi.shape[0]), df_cover_fi['cover'], height=0.6, color='Salmon')
    for i, var in enumerate(df_cover_fi['var_name']):
        plt.annotate(var, xy=(0.001, i), va='center', ha='left', fontsize=4, color='black', fontweight='normal')
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.7)
    plt.xticks(fontsize=5)
    plt.yticks([])
    plt.xlabel('sample coverage', fontsize=5)
    plt.title(title3, fontsize=6)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=2.0)

    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'lgbm_feature_importance.png'), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()

    return df_fi


def save_model(model, model_type, save_path):
    """
    保存模型，若是logistics回归模型，则将模型的参数存成csv文件，若是lgbm或xgboost模型，则存为pickle文件
    Parameters
    ----------
    model: object
        训练好的模型
    model_type: str, option['lr', 'gbm']
        模型类型
    save_path: str
        结果保存路径
    """
    if model_type == 'lr':
        params = list(zip(model.params.index, model.params.values))
        modelfile = pd.DataFrame(params, columns=['var_name', 'coefficient'])
        modelfile.to_csv(save_path, index=False)

    elif model_type == 'gbm':
        if os.path.isfile(save_path) == False:
            os.mknod(save_path)
        with open(save_path, 'wb+') as f:
            pickle.dump(model, f, -1)
        return True

    else:
        raise ValueError('model_type can only be "lr" or "gbm"')


def pickle_load(path):
    """
    载入pickle文件
    Parameters
    ----------
    path: str
        pickle文件路径

    Returns
    -------
    obj: object
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    return obj
