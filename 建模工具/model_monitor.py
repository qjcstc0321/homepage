# coding=utf8
# Author: Jingcheng Qiu

"""
模型监控工具包
"""


import os
import sys
from datetime import datetime, timedelta
import ast
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib


# 绘图中显示中文
if sys.version_info.major == 2:
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.setdefaultencoding('utf-8')
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
try:
    __currdir = os.path.dirname(os.path.abspath(__file__))
    zhfont = matplotlib.font_manager.FontProperties(fname=__currdir + '/ukai.ttc')
except:
    print('Cannot find ukai.ttc')
    zhfont = matplotlib.font_manager.FontProperties()
plt.switch_backend('agg')


def roll_display(df, show_rows, keep_rows=0):
    '''
    滚动显示DataFrame
    Args:
        df: DataFrame
        show_rows: int, 要显示的行数
        keep_rows: int, 顶部要保留的行数
    Return:
        df_return: DataFrame
    Raise:
        error information
    '''
    base = df.head(keep_rows)  #保留第前几行

    #滚动显示m行
    if df.shape[0] > show_rows :
        df_return = df[-show_rows:].copy()
    else:
        df_return = df[keep_rows:].copy()
    df_return = pd.concat([base, df_return], ignore_index=True)

    return df_return

def calc_kl(p, q):
    '''
    计算离散概率分布P、Q的KL散度
    Args:
        p: array, 概率分布1
        q: array, 概率分布2
    Return:
        kl: Float, 分布P相对于分布Q的KL散度，即KL(P||Q)
    Raise:
        ValueError
    '''
    if len(p) != len(q):
        raise ValueError('p and q must have same length')
    kl = np.sum(p * np.log(p/q))

    return kl

def calc_psi(delp, valid):
    '''
    计算PSI
    Args:
        delp: array, 实际概率分布
        valid: array, 期望概率分布
    Return:
        psi: Float
    Raise:
        ValueError
    '''
    psi = calc_kl(delp, valid) + calc_kl(valid, delp)

    return psi

def calc_ks(y_actual, y_predicted):
    '''
    计算模型的KS指标
    Args:
        y_actual: array, target的实际值
        y_predicted: array, target的预测值
    Return:
        ks: Float
    Raise:
        ValueError
    '''
    fpr, tpr, thresholds = roc_curve(y_actual, y_predicted)
    ks = max(tpr - fpr)

    return ks

def calc_rmse(y_actual, y_predicted):
    '''
    计算RMSE
    Args:
        y_actual: array, target的实际值
        y_predicted: array, target的预测值
    Return:
        rmse: Float
    Raise:
        ValueError
    '''
    if len(y_actual) != len(y_predicted):
        raise ValueError('Different length between y_actual and y_predicted!')
    rmse = np.sqrt(np.sum((y_actual - y_predicted) ** 2) / len(y_actual))

    return rmse

def create_monitor_file(monitor_name, output_path, bins=10):
    '''
    创建监控文件模板
    Args:
        monitor_name: string, 监控的类型 可选['score_monitor', 'score_target_monitor', 'var_monitor]
        output_path: string, 文件保存路径
        bins: int, 分数监控分bin的个数
    Raise:
        ValueError
    '''
    if monitor_name == 'score_monitor':
        header = ['date', 'group', 'samples', 'avg_score', 'psi'] + \
                 ['qtl_{0}'.format(i) for i in range(11)] + \
                 ['sbin_{0}'.format(j) for j in range(1, bins+1)]
    elif monitor_name == 'score_target_monitor':
        header = ['date','group','model_bin','samples','target_rate','avg_score','residual','ks','rmse']
    elif monitor_name == 'var_monitor':
        header = ['date','var_name','missrate','mean','std','min','qtl_10','qtl_20','qtl_30','qtl_40','qtl_50',
                  'qtl_60','qtl_70','qtl_80','qtl_90','max','cat_value_pct']
    else:
        raise ValueError('monitor_name is wrong, only allow "score_monitor", "score_target_monitor", "var_monitor".')
    df = pd.DataFrame(columns=header)
    df.to_csv(output_path, index=False, encoding='utf-8')


class ModelMonitor(object):
    '''
    Attributes:
        score_var: string, score变量名
        target: string, target变量名
        group_var: string, 分组的变量名
        groups: list, 组名
        group_cnt: int, 分组个数
        cut_points: list or dict，score切分点
        output_path: string, 监控结果保存地址
    Method:
        score_monitor: 打分监控
        score_monitor_plot: 打分监控绘图
        score_target_monitor: score & target rate监控
        score_target_plot: score & target rate监控绘图
        var_monitor: 变量监控
        var_monitor_plot: 变量监控绘图
    '''
    def __init__(self, score_var='score', target='target', group_var=None, groups=None,
                 cut_points=None, output_path=os.getcwd()+'/monitor'):
        '''
        Args:
            score_var: string, score变量名，默认为'score'
            target: string, target变量名，默认为'target'
            group_var: string, 分组的变量名，若无分组可不填，默认为None
            groups: list, 组名，若无分组可不填，默认为None
            cut_points: list or dict, score切分点，若不同组的切分点不同，需传入一个dict，若不分bin可不填，默认为None
            output_path: string, 监控结果保存地址，默认为'./monitor'
        '''
        self.score_var = score_var
        self.target = target
        self.group_var = group_var
        if (group_var is None) ^ (groups is None):
            raise ValueError('group_var和groups必须同时填入!')

        if groups is None:
            self.groups = ['all']
        else:
            self.groups = groups
        self.group_cnt = len(self.groups)

        self.output_path = output_path
        if os.path.exists(self.output_path) == False:
            os.makedirs(self.output_path)

        if cut_points is None:
            self.cut_points = {'all':[0.0, 1.0]}
        elif type(cut_points) == list:
            self.cut_points = {'all': cut_points}
        else:
            self.cut_points = cut_points

    def score_monitor(self, df_score, valid=np.full(10, 0.1), date=datetime.strftime(datetime.now(), '%Y-%m-%d')):
        '''
        模型每日新样本打分监控，对应文件score_distribution_monitor_daily.csv
        1.每日打分样本数和平均分监控
        2.每日的打分的0.0-1.0共11个分位点监控
        3.根据切分点分bin后每个bin中样本的占比监控
        Args:
            df_score: DataFrame, 每日打分的结果
            valid: 样本在训练集的分布占比， 默认为10%
            date: string, 日期
        Return:
            None
        '''
        # 读取监控文件
        if os.path.isfile(self.output_path + '/score_distribution_monitor_daily.csv') == False:
            create_monitor_file('score_monitor', output_path=self.output_path + '/score_distribution_monitor_daily.csv')
        df_monitor_daily = pd.read_csv(self.output_path + '/score_distribution_monitor_daily.csv')

        # 确定开始写入的行数
        if date in df_monitor_daily['date'].unique():
            n = df_monitor_daily[df_monitor_daily['date'] == date].index.min()
        else:
            n = df_monitor_daily.shape[0]

        for i, group in enumerate(self.groups):
            if group == 'all':
                df_group = df_score.copy()
            else:
                df_group = df_score[df_score[self.group_var] == group].copy().reset_index(drop=True)
            group_cut_points = self.cut_points[group]       # 分bin切分点
            group_bin_cnt = len(group_cut_points) - 1       # 分bin个数

            # 给打分结果分bin
            df_group['model_bin'] = pd.cut(df_group[self.score_var], bins=group_cut_points,
                                           precision=6, labels=list(range(1, group_bin_cnt+1)))

            # 计算监控指标
            df_monitor_daily.loc[n+i, 'date'] = date
            df_monitor_daily.loc[n+i, 'group'] = group
            df_monitor_daily.loc[n+i, 'samples'] = df_group.shape[0]
            df_monitor_daily.loc[n+i, 'avg_score'] = df_group[self.score_var].mean()
            for j in range(11):
                df_monitor_daily.loc[n + i, 'qtl_{0}'.format(j)] = df_group[self.score_var].quantile(float(j) / 10)
            for k in range(1, group_bin_cnt+1):
                df_monitor_daily.loc[n+i, 'sbin_{0}'.format(k)] = (df_group['model_bin'] == k).mean()
            col_tmp = ['sbin_{0}'.format(m) for m in range(1, group_bin_cnt+1)]
            df_monitor_daily.loc[n+i, 'psi'] = calc_psi(delp=np.float64(df_monitor_daily.loc[n+i, col_tmp].values),
                                                        valid=valid)

        # 保存监控文件
        df_monitor_daily.to_csv(self.output_path+'/score_distribution_monitor_daily.csv', index=False, encoding='utf-8')

    def score_monitor_plot(self, show_days, keep_base=False):
        '''
        模型每日新样本打分监控画图，对应图片score_distribution_monitor_graph.png
        1.第一张图监控每日打分样本的个数和0%—100%分位数值的变化
        2.第二张图监控每个bin中样本占比的变化
        Args:
            show_days: int, 展示的天数
            keep_base: bool, 是否展示base记录
        Return:
            None
        '''
        # 读取监控文件
        df_monitor_daily = pd.read_csv(self.output_path + '/score_distribution_monitor_daily.csv')

        # 滚动显示部分天数
        if keep_base:
            df_monitor_daily = roll_display(df_monitor_daily, show_rows=show_days * self.group_cnt,
                                            keep_rows=self.group_cnt)
        else:
            df_monitor_daily = roll_display(df_monitor_daily, show_rows=show_days * self.group_cnt, keep_rows=0)

        # 配置绘图参数
        xticks_list = list(df_monitor_daily['date'].unique())   # 日期坐标
        rows = len(xticks_list)     # 数据长度

        # 画图
        plt.subplots(self.group_cnt, 2, figsize=(14, 5*self.group_cnt), dpi=150)
        for i, group in enumerate(self.groups):
            df_group_monitor = df_monitor_daily[df_monitor_daily['group'] == group].reset_index(drop=True)
            group_bin_cnt = len(self.cut_points[group]) - 1      # 分bin的个数

            # sample count监控
            plt.subplot(self.group_cnt, 2, 2*i+1)
            plt.bar(left=range(rows), height=df_group_monitor['samples'],
                    width=.3, color='limegreen', label='Sample cnt')
            plt.axis(ymin=0, ymax=1.8 * df_group_monitor['samples'].max())
            plt.xticks(range(rows), xticks_list, rotation=90, fontsize=10)
            plt.legend(loc=2, fontsize=12)
            plt.twinx()
            # quantile score监控
            line_colors = plt.cm.Blues(np.linspace(0.3, 1, 11))
            for j, color in enumerate(line_colors):
                plt.plot(range(rows), df_group_monitor['qtl_{0}'.format(j)], color=color,
                         label='{0}% Quantile'.format(10*j))
            plt.axis(ymin=0, ymax=1.2)
            plt.legend(loc=1, fontsize=6)
            plt.title('Quantile Score Monitor of {0}'.format(group), fontsize=14)

            # score bin监控
            plt.subplot(self.group_cnt, 2, 2*i+2)
            bar_colors = plt.cm.Blues(np.linspace(0.3, 1, group_bin_cnt))
            y_offset = df_group_monitor['sbin_1']
            for k, color in enumerate(bar_colors):
                if k == 0:
                    plt.bar(range(rows), height=df_group_monitor['sbin_{0}'.format(k+1)], width=.3, color=color,
                            label='Bin{0} proportion'.format(k+1))
                else:
                    plt.bar(range(rows), height=df_group_monitor['sbin_{0}'.format(k+1)], bottom=y_offset, width=.3,
                            color=color, label='Bin{0} proportion'.format(k+1))
                    y_offset += df_group_monitor['sbin_{0}'.format(k+1)]
                plt.plot(range(rows), y_offset, color=color)
            plt.axis(ymin=0.0, ymax=1.2)
            plt.xticks(range(rows), xticks_list, rotation=90, fontsize=10)
            plt.yticks(np.arange(0, 1.2, 0.2), ['0', '20%', '40%', '60%', '80%', '100%'])
            psi_today = round(df_group_monitor['psi'].iloc[-1], 3)
            plt.title('Equal Score Population Monitor of {0}, PSI={1}'.format(group, psi_today), fontsize=14)

        # 保存图片
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.savefig(self.output_path+'/score_distribution_monitor_graph.png', bbox_inches='tight')
        plt.close()

    def score_target_monitor(self, df_score_target, lag_days, interval_days,
                             date=datetime.strftime(datetime.now(), '%Y-%m-%d')):
        '''
        模型回溯score和target rate监控，对应文件score_target_monitor_daily.csv
        1.根据target的观测滞后时间设置lag_days，
        2.监控每个bin中的target rate、平均分、KS及target rate和平均分的差值
        Args:
            df_score_target: DataFrame, 包含打分和target的结果
            lag_days: int, target观测的滞后天数
            interval_days: int, 观测间隔天数
            date: string, 日期
        Return:
            None
        '''
        # 读取历史监控记录文件
        if os.path.isfile(self.output_path + '/score_target_monitor_daily.csv') == False:
            create_monitor_file('score_target_monitor',
                                output_path=self.output_path + '/score_target_monitor_daily.csv')
        df_monitor_daily = pd.read_csv(self.output_path+'/score_target_monitor_daily.csv')

        # 生成监控日期
        end_date = datetime.strftime(datetime.strptime(date, '%Y-%m-%d') - timedelta(lag_days),
                                     '%Y-%m-%d')  # 观测截止日期
        begin_date = datetime.strftime(datetime.strptime(end_date, '%Y-%m-%d') - timedelta(interval_days-1),
                                       '%Y-%m-%d')  # 观测开始日期
        if begin_date == end_date:
            dates = begin_date
        else:
            dates = begin_date + ' ~ ' + end_date  # 观测日期区间

        # 确定开始写入的行数
        if dates in df_monitor_daily['date'].unique():
            n = df_monitor_daily[df_monitor_daily['date'] == dates].index.min()
        else:
            n = df_monitor_daily.shape[0]

        for i, group in enumerate(self.groups):
            if group == 'all':
                df_group = df_score_target.copy()
            else:
                df_group = df_score_target[df_score_target[self.group_var] == group].reset_index(drop=True)
            group_cut_points = self.cut_points[group]  # 分bin切分点
            group_bin_cnt = len(group_cut_points) - 1  # 分bin个数
            if group_bin_cnt == 1:
                bins = [1, 'total']
            else:
                bins = [k for k in range(1, group_bin_cnt+1)] + ['total']

            # 给打分结果分bin
            df_group['model_bin'] = pd.cut(df_group[self.score_var], bins=group_cut_points,
                                           precision=5, labels=list(range(1, group_bin_cnt+1)))

            # 计算监控指标
            for j, bin in enumerate(bins):
                if bin == 'total':
                    df_part = df_group
                else:
                    df_part = df_group[df_group['model_bin'] == bin]
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'date'] = dates
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'group'] = group
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'model_bin'] = str(bin)
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'samples'] = df_part.shape[0]
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'target_rate'] = df_part[self.target].mean()
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'avg_score'] = df_part[self.score_var].mean()
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'residual'] = abs(df_part[self.score_var].mean() - df_part[self.target].mean())
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'ks'] = calc_ks(df_part[self.target], df_part[self.score_var])
                df_monitor_daily.loc[n + (group_bin_cnt+1)*i + j, 'rmse'] = np.nan
            target_rate = df_monitor_daily.loc[n:(n + (group_bin_cnt+1)*i + group_bin_cnt), 'target_rate']
            avg_score = df_monitor_daily.loc[n:(n + (group_bin_cnt+1)*i + group_bin_cnt), 'avg_score']
            df_monitor_daily.loc[n + (group_bin_cnt+1)*i + group_bin_cnt, 'rmse'] = calc_rmse(target_rate, avg_score)

        # 保存监控文件
        df_monitor_daily.to_csv(self.output_path+'/score_target_monitor_daily.csv', index=False, encoding='utf8')

    def score_target_plot(self, show_days, lag_days, interval_days, keep_base=False,
                          date=datetime.strftime(datetime.now(), '%Y-%m-%d')):
        '''
        模型每日新样本打分监控画图，对应文件score_target_monitor_graph.png
        1.第一张图监控每个bin中样本的target rate变化
        2.第二张图模型的sloping
        Args:
            show_days: int, 展示的天数
            lag_days: int, target观测的滞后天数
            interval_days: int, 观测间隔天数
            keep_base: bool, 是否展示base记录
            date: string, 日期
        Return:
            None
        '''
        # 读取历史监控文件
        df_monitor_daily = pd.read_csv(self.output_path + '/score_target_monitor_daily.csv')

        # 配置绘图参数
        end_date = datetime.strftime(datetime.strptime(date, '%Y-%m-%d') - timedelta(lag_days),
                                     '%Y-%m-%d')  # 观测截止日期
        begin_date = datetime.strftime(datetime.strptime(end_date, '%Y-%m-%d') - timedelta(interval_days-1),
                                       '%Y-%m-%d')  # 观测开始日期
        if begin_date == end_date:
            dates = begin_date
        else:
            dates = begin_date + ' ~ ' + end_date  # 观测日期区间

        # 绘图
        plt.subplots(self.group_cnt, 2, figsize=(14, 5*self.group_cnt), dpi=150)
        for i, group in enumerate(self.groups):
            df_group_monitor = df_monitor_daily[df_monitor_daily['group'] == group].copy().reset_index(drop=True)
            group_bin_cnt = len(self.cut_points[group]) - 1  # 分bin的个数
            colors = plt.cm.Blues(np.linspace(0.3, 1, group_bin_cnt))  # 颜色

            # 滚动显示部分天数
            if keep_base:
                df_group_monitor = roll_display(df_group_monitor, show_rows=show_days * (group_bin_cnt+1),
                                                keep_rows=group_bin_cnt+1)
            else:
                df_group_monitor = roll_display(df_group_monitor, show_rows=show_days * (group_bin_cnt+1), keep_rows=0)

            xticks_list = df_group_monitor['date'].unique()  # 日期坐标
            rows = len(xticks_list)  # 数据长度

            # 分bin target监控
            plt.subplot(self.group_cnt, 2, 2*i+1)
            for j, color in enumerate(colors):
                df_bin_monitor = df_group_monitor[df_group_monitor['model_bin'] == str(j+1)]
                plt.plot(range(rows), df_bin_monitor['target_rate'], color=color,
                         linestyle='-', marker='.', label='bin {0}'.format(j+1))
            plt.axis(ymin=0, ymax=1.2)
            plt.xticks(range(rows), xticks_list, rotation=90, fontsize=8)
            plt.yticks(np.arange(0, 1.2, 0.2), ['0', '20%', '40%', '60%', '80%', '100%'])
            plt.ylabel('target rate', fontsize=13)
            plt.legend(loc=1, fontsize=7)
            plt.title('Equal Score bin Target rate Monitor of {0}'.format(group), fontsize=14)

            df_monitor_part = df_group_monitor[(df_group_monitor['date'] == dates) & \
                                               (df_group_monitor['model_bin'] != 'total')].copy()
            df_monitor_part['model_bin'] = df_monitor_part['model_bin'].astype(np.int64)
            df_monitor_part = df_monitor_part.sort_values(by='model_bin').reset_index(drop=True)
            max_residual_index = df_monitor_part['residual'].idxmax(axis=0)

            # 最新一个日期的sloping监控
            plt.subplot(self.group_cnt, 2, 2*i+2)
            plt.plot(range(group_bin_cnt), df_monitor_part['avg_score'], color='blue',
                     linestyle='-', marker='.', markersize=5, label='Prediction')
            plt.plot(range(group_bin_cnt), df_monitor_part['target_rate'], color='red',
                     linestyle='-', marker='.', markersize=5, label='Truth')
            plt.plot([max_residual_index, max_residual_index],
                     [df_monitor_part['avg_score'][max_residual_index], df_monitor_part['target_rate'][max_residual_index]],
                     linewidth=1.5, color='g', linestyle='--')
            plt.annotate('max gap: %.2f' % df_monitor_part['residual'].max(),
                         xy=(max_residual_index, df_monitor_part['target_rate'][max_residual_index] - 0.03),
                         xytext=(max_residual_index + 0.5, df_monitor_part['target_rate'][max_residual_index] + 0.05),
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=4.0, shrink=0.05))
            plt.axis(ymin=0, ymax=1.0)
            plt.xticks(range(group_bin_cnt), range(1, group_bin_cnt+1), fontsize=8)
            plt.yticks(np.arange(0, 1.2, 0.2), ['0', '20%', '40%', '60%', '80%', '100%'], fontsize=8)
            plt.xlabel('score bin', fontsize=13)
            plt.ylabel('target rate&avg score', fontsize=13)
            plt.legend(loc=1, fontsize=10)
            plt.title('Sloping Monitor of {0}'.format(group), fontsize=14)

        # 保存图片
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.savefig(self.output_path+'/score_target_monitor_graph.png', bbox_inches='tight')
        plt.close()

    def ks_monitor_plot(self, show_days, keep_base=False):
        '''
        监控模型的KS及Preidict与Truth的RMSE变化，对应文件ks_monitor_graph.png
        Args:
            show_days: int, 展示的天数
            keep_base: bool, 是否展示base记录
        Return:
            None
        '''
        # 读取历史监控文件
        df_monitor_daily = pd.read_csv(self.output_path + '/score_target_monitor_daily.csv')
        df_monitor_total = df_monitor_daily[df_monitor_daily['model_bin'] == 'total'].copy().reset_index(drop=True)

        # 滚动显示部分天数
        if keep_base:
            df_monitor_total = roll_display(df_monitor_total, show_rows=show_days*self.group_cnt,
                                            keep_rows=self.group_cnt)
        else:
            df_monitor_total = roll_display(df_monitor_total, show_rows=show_days*self.group_cnt, keep_rows=0)

        # 配置绘图参数
        xticks_list = df_monitor_total['date'].unique()     # 日期坐标
        rows = len(xticks_list)     # 数据长度
        bar_colors = ['red', 'blue', 'green']       # 柱状图颜色
        line_colors = ['salmon', 'c', 'lightgreen']     # 折线图颜色

        # 画图
        plt.subplots(1, self.group_cnt, figsize=(8, 5), dpi=150)
        plt.subplot(1, 1, 1)
        # sample count监控柱状图
        for i, group in enumerate(self.groups):
            df_group_monitor = df_monitor_total[df_monitor_total['group'] == group].reset_index(drop=True)
            if df_group_monitor.shape[0] != rows:
                raise ValueError('{0} 数据长度不正确: {1} != {2}'.format(group, df_group_monitor.shape[0], rows))
            plt.bar(left=np.arange(rows)-0.2*i, height=df_group_monitor['rmse'], color=bar_colors[i], width=.2,
                    label='{0}'.format(group))
        plt.ylabel('RMSE', fontsize=13)
        plt.xticks(np.arange(len(xticks_list)), xticks_list, rotation=90, fontsize=8)
        plt.legend(loc=2, fontsize=10)
        plt.twinx()

        # ks监控折线图
        for i, group in enumerate(self.groups):
            df_group_monitor = df_monitor_total[df_monitor_total['group'] == group].reset_index(drop=True)
            plt.plot(range(rows), df_group_monitor['ks'], color=line_colors[i], linestyle='-', marker='.',
                     label='{0}'.format(group))
        plt.axis(ymin=0, ymax=1.0)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel('KS', fontsize=13)
        plt.legend(loc=1, fontsize=10)
        plt.title('KS & Sloping RMSE Monitor', fontsize=14)

        # 保存图片
        plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0)
        plt.savefig(self.output_path + '/ks_monitor_graph.png', bbox_inches='tight')
        plt.close()

    def __pct_to_json(self, df, var_name, value_list):
        '''
        string/indicator变量生成json字符串
        '''
        pct_list = []
        for value in value_list:
            if value == 'missing':
                pct = float(df[var_name].isnull().sum() + (df[var_name] == '').sum()) / \
                      df.shape[0]
                pct_list.append(pct)
            else:
                pct = float((df[var_name] == value).sum()) / df.shape[0]
                pct_list.append(pct)
        str_result = "{'value':%s,'pct':%s}" % ("['" + "','".join(value_list) + "']", str(pct_list))

        return str_result

    def __json_to_df(self, df_var_monitor):
        '''
        根据json字符串生成DataFrame
        '''
        df_return = pd.DataFrame(columns=['date', 'value', 'pct'])
        try:
            for dt, value_pct in zip(df_var_monitor['date'].values, df_var_monitor['cat_value_pct'].values):
                df_tmp = pd.DataFrame.from_dict(ast.literal_eval(value_pct))
                df_tmp['date'] = dt
                df_return = pd.concat((df_return, df_tmp), axis=0)
        except:
            raise Exception('"{0}"不是indicator或categorical类型的变量'.format(df_var_monitor['var_name'].iloc[0]))
        df_return = df_return.sort_values(by='date')

        return df_return

    def __grouping_var_list(self, var_list, num=6):
        '''
        生成指定格式的list
        '''
        input_list_result = []
        input_list_part = []
        var_cnt = len(var_list)
        for i in range(var_cnt):
            if i == 0:
                input_list_part.append(var_list[i])
            elif ((i + 1) % num > 0) & ((i + 1) != var_cnt):
                input_list_part.append(var_list[i])
            elif ((i + 1) % num > 0) & ((i + 1) == var_cnt):
                input_list_part.append(var_list[i])
                input_list_result.append(input_list_part)
            elif (i + 1) % num == 0:
                input_list_part.append(var_list[i])
                input_list_result.append(input_list_part)
                input_list_part = []

        return input_list_result

    def var_monitor(self, df_orig, file_name, var_list, cat_value_dict, date=datetime.strftime(datetime.now(), '%Y-%m-%d')):
        '''
        变量监控:
        1.numerical变量监控missing rate、均值和5个主要分位点的值
        2.categorical变量监控cat_value_dict中指定的取值的占比
        Args:
            df_orig: DataFrame, 变量的原始表
            file_name: string, 监控文件名
            var_list: dict, 监控变量的字典, eg{'numerical':[], 'categorical':[], ''indicator':[]}
            cat_value_dict: dict, categorical变量的取值字典，e.g. {"cmstr_idnm_gen":['missing','男','女']}
            date: string, 日期
        Return:
            None
        '''
        # 读取监控文件
        if os.path.isfile(self.output_path + '/{0}'.format(file_name)) == False:
            create_monitor_file('var_monitor', output_path=self.output_path + '/{0}'.format(file_name))
        df_var_monitor = pd.read_csv(self.output_path + '/{0}'.format(file_name))

        n = df_var_monitor.shape[0]  # 开始写入的行数
        df_var_monitor = df_var_monitor[df_var_monitor['date'] != date]  # 如果有今天的数据，则删除

        # numerical变量监控
        for var in var_list['numerical']:
            df_var_monitor.loc[n, 'date'] = date
            df_var_monitor.loc[n, 'var_name'] = var
            df_var_monitor.loc[n, 'missrate'] = (float(df_orig[var].isnull().sum()) + (df_orig[var] == -1.0).sum()) / df_orig.shape[0]
            df_var_monitor.loc[n, 'mean'] = df_orig[var].mean()
            df_var_monitor.loc[n, 'std'] = df_orig[var].std()
            df_var_monitor.loc[n, 'min'] = df_orig[var].min()
            for q in range(1, 10):
                df_var_monitor.loc[n, 'qtl_{0}0'.format(q)] = df_orig[var].quantile(q/10.0)
            df_var_monitor.loc[n, 'max'] = df_orig[var].max()
            n += 1

        # indicator变量
        for var in var_list['indicator']:
            df_var_monitor.loc[n, 'date'] = date
            df_var_monitor.loc[n, 'var_name'] = var
            df_var_monitor.loc[n, 'missrate'] = float(df_orig[var].isnull().sum()) / df_orig.shape[0]
            df_var_monitor.loc[n, 'cat_value_pct'] = "{'value':['missing','0','1'],'pct':[%f,%f,%f]}" % (\
                    float(df_orig[var].isnull().sum()) / df_orig.shape[0],\
                    float(sum(df_orig[var] == 0)) / df_orig.shape[0],\
                    float(sum(df_orig[var] == 1)) / df_orig.shape[0])
            n += 1

        # categorical变量
        for var in var_list['categorical']:
            df_var_monitor.loc[n, 'date'] = date
            df_var_monitor.loc[n, 'var_name'] = var
            df_var_monitor.loc[n, 'missrate'] = float(df_orig[var].isnull().sum() + (df_orig[var] == '').sum()) / df_orig.shape[0]
            df_var_monitor.loc[n, 'cat_value_pct'] = self.__pct_to_json(df_orig, var_name=var, value_list=cat_value_dict[var])
            n += 1

        #保存监控
        df_var_monitor = df_var_monitor.sort_values(by='date')
        df_var_monitor.to_csv(self.output_path + '/{0}'.format(file_name), index=False, encoding='utf-8')

    def var_monitor_plot(self, file_name, show_days, var_list, cat_value_dict, keep_base=False, num=6, suffix=''):
        '''
        变量监控画图
        Args:
            file_name: string, 监控文件名
            show_days: int, 展示的天数
            var_list: dict, 监控变量的字典, e.g. {'numerical':[], 'categorical':[], ''indicator':[]}
            cat_value_dict: dict, categorical变量需要监控的值，e.g. {"cmstr_idnm_gen":['missing','男','女']}
            keep_base: bool, 是否展示base记录
            num: int, 一张图上变量的个数，默认为6个
            suffix: string, 保存文件名时使用，用户群名称，若无分组则不填
        Return:
            None
        '''

        # 读取监控文件
        df_var_monitor = pd.read_csv(self.output_path + '/{0}'.format(file_name))
        var_list_total = var_list['numerical'] + var_list['categorical'] + var_list['indicator']
        var_cnt = len(var_list_total)

        # 滚动显示部分天数
        if keep_base:
            df_var_monitor = roll_display(df_var_monitor, show_rows=show_days * var_cnt, keep_rows=var_cnt)
        else:
            df_var_monitor = roll_display(df_var_monitor, show_rows=show_days * var_cnt, keep_rows=0)

        # 配置绘图参数
        var_group_list = self.__grouping_var_list(var_list_total, num=num)
        xticks_list = [date[5:10] for date in df_var_monitor['date'].unique()]  # 日期坐标
        rows = len(xticks_list)
        cols = ['qtl_10', 'qtl_30', 'qtl_50', 'qtl_70', 'qtl_90']

        # 画图
        for i, var_list_part in enumerate(var_group_list):
            subplot_row = (len(var_list_part) + 1) // 2     # 画布的行数
            plt.subplots(subplot_row, 2, figsize=(14, 5*subplot_row), dpi=150)

            # 画布中每个变量循环
            for j, var in enumerate(var_list_part):
                df_var_monitor_tmp = df_var_monitor.loc[df_var_monitor['var_name'] == var, :].reset_index(drop=True)
                plt.subplot(subplot_row, 2, j+1)
                # numerical单变量画图
                if var in var_list['numerical']:
                    colors = plt.cm.Blues(np.linspace(0.3, 1, 5))
                    plt.bar(left=range(rows), height=df_var_monitor_tmp['missrate'],
                            width=.3, color='r', edgecolor='r', label='Missing rate')
                    plt.axis(ymin=0, ymax=1.2)
                    plt.xticks(range(rows), xticks_list, rotation=90, fontsize=8)
                    plt.yticks(np.arange(0, 1.4, 0.2), ['0', '20%', '40%', '60%', '80%', '100%', ''])
                    plt.legend(loc=2, fontsize=6.5)
                    plt.twinx()
                    plt.plot(range(rows), df_var_monitor_tmp['mean'], color='b', linestyle='dashed', label='mean')
                    for col, color in zip(cols, colors):
                        plt.plot(range(rows), df_var_monitor_tmp[col], color=color, label=col)
                    plt.axis(ymin=0, ymax=1.6 * df_var_monitor_tmp['qtl_90'].max())
                    plt.legend(loc=1, fontsize=6.5)
                    plt.title(var, fontsize=15)

                # categorical/indicator 单变量画图
                elif (var in var_list['categorical']) | (var in var_list['indicator']):
                    # 将文件中json串转为df
                    df_plot_tmp = self.__json_to_df(df_var_monitor_tmp, var_name=var)
                    if var in var_list['categorical']:
                        bar_layer_list = cat_value_dict[var]
                    elif var in var_list['indicator']:
                        bar_layer_list = ['missing', '0', '1']
                    colors = plt.cm.Paired(np.linspace(0.1, 1, len(bar_layer_list)))
                    y_offset = df_plot_tmp['pct'][df_plot_tmp['value'] == bar_layer_list[0]].reset_index(drop=True)
                    plot_list = []

                    # 对每个变量中不同的value值循环
                    for k, value in enumerate(bar_layer_list):
                        # missing值使用红色
                        if value == 'missing':
                            plot_list.append(plt.bar(left=range(rows),
                                                     height=df_plot_tmp['pct'][df_plot_tmp['value'] == value],
                                                     width=.3, color='r'))
                        elif (k == 0) & (value != 'missing'):
                            plot_list.append(plt.bar(left=range(rows),
                                                     height=df_plot_tmp['pct'][df_plot_tmp['value'] == value],
                                                     width=.3, color=colors[k]))
                        else:
                            plot_list.append(plt.bar(left=range(rows),
                                                     height=df_plot_tmp['pct'][df_plot_tmp['value'] == value],
                                                     bottom=y_offset, width=.3, color=colors[k]))
                            y_offset += df_plot_tmp['pct'][df_plot_tmp['value'] == bar_layer_list[k]].reset_index(
                                drop=True)
                    plt.legend(plot_list, [v.decode('utf-8') for v in bar_layer_list], loc=1, fontsize=6.5, prop=zhfont)
                    plt.axis(ymin=0, ymax=1.2)
                    plt.xticks(range(rows), xticks_list, rotation=90, fontsize=9)
                    plt.yticks(np.arange(0, 1.2, 0.2), ['0', '20%', '40%', '60%', '80%', '100%'])
                    plt.title(var, fontsize=15)

            # 保存图片
            plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
            plt.savefig(self.output_path + '/var_monitor_graph_{0}_{1}.png'.format(suffix, i), bbox_inches='tight')
            plt.close()
