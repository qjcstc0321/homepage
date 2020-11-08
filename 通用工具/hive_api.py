# coding: utf-8
# Author: Jingcheng Qiu

"""
hive和impala读写工具
"""


import os
from time import sleep
from datetime import datetime
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
from hdfs import InsecureClient


def sql_request(sql_file):
    """
    读取文件生成SQL代码
    Parameters
    ----------
    sql_file: str
        SQL文件路径

    Returns
    -------
    sql_string: str
        解析好的SQL语句
    """
    f = open(sql_file)
    try:
        sql_string = f.read()
    except:
        raise Exception
    finally:
        f.close()
    
    return sql_string


def connect_impala(user, password, port='impala'):
    """
    连接impala
    Parameters
    ----------
    user: str
        Hue账号
    password: str
        Hue密码
    port: str, options ['impala', 'hive'], default 'impala'
        查询端口，可选impala或hive

    Returns
    -------
    conn: object
        连接对象
    """
    if port == 'impala':
        host = '10.9.8.42'    # 备选ip ['10.9.8.38', '10.9.8.210']
        port = 21050  
    elif port == 'hive':
        host = '10.9.8.111'
        port = 10000
    else:
        raise Exception('Port only choose "impala" or "hive"!')
    
    try:
        print('Use ip: {0}, port: {1}'.format(host, port))
        conn = connect(host=host, auth_mechanism='PLAIN', port=port, user=user, password=password)
    except:
        raise Exception('Connection failed, please check your userid or password!')
        
    return conn


def sql_query(sql_string, user, password, sql_params=None, trytimes=1, port='impala'):
    """
    执行SQL语句
    Parameters
    ----------
    sql_string: str
        一段或多段SQL语句, 每段SQL语句用";"结尾
    user: str
        Hue账号
    password: str
        Hue密码
    sql_params: list
        sql参数
    trytimes: int, default 1
        尝试次数，默认为1次
    port: str, options ['impala', 'hive'], default 'impala'
        查询端口，可选impala或hive
    """
    conn = connect_impala(user=user, password=password, port=port)
    cursor = conn.cursor()
    for sql_part in [s for s in sql_string.split(';') if s.strip() != '']:
        if sql_params is not None:
            sql_part = sql_part.format(*sql_params)
        print(sql_part)
        part_start_time = datetime.now()
        while trytimes > 0:
            try:
                cursor.execute(sql_part)
                break
            except Exception as error:
                trytimes -= 1
                if trytimes == 0:
                    conn.close()
                    raise error
                sleep(60)
        print('Time cost: {0}'.format(datetime.now() - part_start_time))
    conn.close()
    
    return True


def impala_to_df(sql_string, user, password, port='impala'):
    """
    从impala上查询数据并生成dataframe, 会自动将decimal类型转换成float类型。
    Parameters
    ----------
    sql_string: str
        SQL查询语句
    user: str
        Hue账号
    password: str
        Hue密码
    port: str, options ['impala', 'hive'], default 'impala'
        查询端口，可选impala或hive

    Returns
    -------
    query_result: Dataframe
        查询结果
    """
    conn = connect_impala(user=user, password=password, port=port)
    cursor = conn.cursor()
    start_time = datetime.now()
    try:
        cursor.execute(sql_string.split(';')[0])
    except Exception as error:
        raise error

    query_result = as_pandas(cursor)
    conn.close()
    
    if query_result.shape[0] == 0:
        raise Exception('No data, please check your query code!')
    
    # 修正列名和列类型
    col_names = []
    for var in query_result.columns:
        if len(var.split('.')) > 1:
            col_names.append(var.split('.')[-1])
        else:
            col_names.append(var)
        if min(query_result[var].isnull()) == 0:
            if type(query_result[var][query_result[var].isnull() == 0].iloc[0]).__name__ == 'Decimal':
                query_result[var] = query_result[var].astype(float)
            elif type(query_result[var][query_result[var].isnull() == 0].iloc[0]).__name__ == 'Timestamp':
                query_result[var] = query_result[var].astype(str)
    query_result.columns = col_names
    print('Return {0} rows, Time cost: {1}'.format(query_result.shape[0], datetime.now() - start_time))

    return query_result


def create_hdfs_dir(hdfs_path, host='http://10.9.8.120:14000', user='hive'):
    """
    在HDFS创建一个目录
    Parameters
    ----------
    hdfs_path: str
        HDFS路径, eg. '/external/cbd/test'
    host: str, default 'http://10.2.8.52:14000'
        HDFS的域名
    user: str, default 'hive'
        用户名

    Returns
    -------
    response: str
        执行结果
    """
    hdfs_conn = InsecureClient(url=host, user=user)
    response = hdfs_conn.makedirs(hdfs_path)

    return response


def upload_to_hdfs(file_path, hdfs_path, host='http://10.9.8.120:14000', user='hive', overwrite=True):
    """
    从本地上传文件到指定的HDFS目录
    Parameters
    ----------
    file_path: str
        文件的绝对路径 eg. '/home/user/test.csv'
    hdfs_path: str
        HDFS路径, eg. '/external/cbd/test'
    host: str, default 'http://10.2.8.52:14000'
        HDFS的域名
    user: str, default 'hive'
        用户名
    overwrite: bool, default True
        若该文件已存在，是否覆盖

    Returns
    -------
    response: str
        执行结果
    """
    hdfs_conn = InsecureClient(url=host, user=user)
    response = hdfs_conn.upload(hdfs_path, file_path, overwrite=overwrite)

    return response


def download_from_hdfs(hdfs_path, save_path=None, host='http://10.9.8.120:14000', user='hive', overwrite=True):
    """
    从HDFS中下载文件
    Parameters
    ----------
    hdfs_path: str
        HDFS路径
    save_path: str, default None
        本地存放文件的路径，若不填则会在当前工作目录下创建一个tmp_folder的文件夹用于存放文件
    host: str, default 'http://10.9.8.120:14000'
        HDFS的域名
    user: str, default 'hive'
        用户名
    overwrite: bool, default True
        若存放路径下已有文件，是否覆盖
    """
    if save_path is None:
        save_path = os.path.join(os.getcwd(), 'tmp_folder')     # 存放文件的临时目录

    # 建立存放文件的文件夹，若文件夹已存在则清空其中的文件
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        if overwrite:
            for f in os.listdir(save_path):
                os.remove(os.path.join(save_path, f))

    # 从HDFS目录下载文件
    hdfs_conn = InsecureClient(url=host, user=user)
    if hdfs_conn.status(hdfs_path)['type'] == 'FILE':
        hdfs_conn.download(hdfs_path, os.path.join(save_path, os.path.basename(hdfs_path)))
    else:
        start_time = datetime.now()
        cnt = 0
        for f in hdfs_conn.list(hdfs_path):
            if hdfs_conn.status(os.path.join(hdfs_path, f))['type'] == 'FILE':
                hdfs_conn.download(os.path.join(hdfs_path, f), os.path.join(save_path, f))
                cnt += 1
        print('Successfully Downloaded {0} files, Time cost: {1}'.format(cnt, datetime.now() - start_time))

    return 'Download completed'


def recursion_delete_hdfs(hdfs_path, host='http://10.9.8.120:14000', user='hive'):
    """
    递归删除HDFS文件夹
    Parameters
    ----------
    hdfs_path: str
        待删除的HDFS路径
    host: str, default 'http://10.9.8.120:14000'
        HDFS的域名
    user: str, default 'hive'
        用户名
    """
    hdfs_conn = InsecureClient(url=host, user=user)
    # print('是否确认删除文件(夹): {}?'.format(hdfs_path))
    # response = input()
    # if response != '':
    #     return
    if hdfs_conn.status(hdfs_path)['type'] == 'FILE':
        print('Delete file:', hdfs_path)
        hdfs_conn.delete(hdfs_path)
    elif hdfs_conn.status(hdfs_path)['type'] == 'DIRECTORY':
        if len(hdfs_conn.list(hdfs_path)) == 0:
            print('Delete path: ', hdfs_path)
            hdfs_conn.delete(hdfs_path)
        else:
            for p in hdfs_conn.list(hdfs_path):
                nextpath = os.path.join(hdfs_path, p)
                recursion_delete_hdfs(hdfs_path=nextpath, host=host, user=user)
            print('Delete path: ', hdfs_path)
            hdfs_conn.delete(hdfs_path)
    else:
        raise ValueError('unknown path type: {}'.format(hdfs_conn.status(hdfs_path)['type']))


def add_partition(table_name, hdfs_path, user, password, date=datetime.strftime(datetime.now(), '%Y-%m-%d')):
    """
    添加分区
    Parameters
    ----------
    table_name: str
        待添加分区的表名
    hdfs_path: str
        存放分区数据的HDFS路径
    user: str
        Hue账号
    password: str
        Hue密码
    date: str, default now
        添加分区的日期
    """
    sql_string = '''alter table {0} drop if exists partition(dt="{1}");
alter table {0} add partition(dt="{1}") location "{2}/dt={1}";'''.format(table_name, date, hdfs_path)
    sql_query(sql_string, user=user, password=password, port='hive')


def create_external_table(df, table_name, hdfs_path, user, password, col_list=None, port='impala'):
    """
    根据DataFrame在impala上生成外部表
    Parameters
    ----------
    df: DataFrame
    table_name: str
        表名
    hdfs_path: str
        存放数据的HDFS路径
    user: str
        Hue账号
    password: str
        Hue密码
    col_list: list, default None
        列名与列类型的二维列表 eg.[['user_id', 'bigint'], ['owingamount', 'double']
        若不指定则默认为dataframe的列名和列类型。
    port: str, options ['impala', 'hive'], default 'impala'
        执行端口，可选impala或hive
    """
    filename = '{0}.csv'.format(table_name.split('.')[1])
    df.to_csv(filename, header=False, index=False)
    file_path = os.getcwd() + '/' + filename
    create_hdfs_dir(hdfs_path, user=user)
    upload_to_hdfs(file_path=file_path, hdfs_path=hdfs_path, user=user)
    os.remove(file_path)
    
    if col_list is None:
        col_name = list(df.columns)
        col_type = []
        for var in col_name:
            if type(df[var][0]) in (int, bool, np.int64):
                col_type.append('INT')
            elif type(df[var][0]) in (float, np.float64):
                col_type.append('FLOAT')
            elif type(df[var][0]) == str:
                col_type.append('STRING')
            else:
                raise Exception('Unknown column type: {0}'.format(var))
        col_list = zip(col_name, col_type)
    col_declare = [i+' '+j for i, j in col_list]
    
    sql_string = '''
    DROP TABLE IF EXISTS {0};
    CREATE EXTERNAL TABLE {0} (
    {1}
    ) row format delimited fields terminated by "," stored as textfile location "{2}";
    '''.format(table_name, ',\n'.join(col_declare), hdfs_path)

    sql_query(sql_string, user=user, password=password, port=port)
    
    if port == 'hive':
        sql_query('INVALIDATE METADATA {0};'.format(table_name), user=user, password=password, port='impala')
        
    return True
