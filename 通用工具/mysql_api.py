# coding: utf-8
# Author: Jingcheng Qiu

"""
mysql写入和查询接口
"""


import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
import pymysql


def connect_mysql(config):
    '''
    创建与mysql数据库的连接
    Args:
        config: dict, 数据库连接参数
    Return:
        conn: object, DBAPI connection
    '''
    url = URL(
        drivername=config['drivername'],
        username=config['username'],
        password=config['password'],
        host=config['host'],
        port=config['port'],
        database=config.get('database') or None,
        query=config.get('query') or {'charset': 'utf8'})

    engine = create_engine(url)
    conn = engine.connect()

    return conn


def __df_to_dict(df, dtypes):
    '''
    将dataframe转成list
    Args:
        df: DataFrame
        dtypes: dict, 数据类型
    '''
    data_list = df.to_dict(orient='records')
    new_data_list = []
    for data in data_list:
        new_data = {}
        for column in dtypes:
            new_data[column] = dtypes[column](data[column])
        new_data_list.append(new_data)

    return new_data_list


def mysql_query(sql, db_config):
    '''
    mysql查询语句，用于执行一些没有返回值的sql语句
    Args:
        sql: string, sql查询语句
    '''
    conn = connect_mysql(db_config)
    conn.execute(sql.split(';')[0])
    conn.close()

    return True


def insert_to_mysql(df, table, dtypes, db_config):
    '''
    往表中插入数据
    Args:
        df: DataFrame
        table: str, 表名
        dtypes: dict, 数据类型 eg.{"age": int, "dt":str}
    '''
    columns = [col for col in dtypes]
    insert_columns = ','.join(columns)
    insert_values = ','.join([':' + col for col in columns])

    sql = '''insert into {table_name} ({insert_columns})
values ({insert_values})
'''.format(table_name=table, insert_columns=insert_columns, insert_values=insert_values)

    data = __df_to_dict(df, dtypes)
    conn = connect_mysql(db_config)
    conn.execute(text(sql), data)

    return True


def mysql_to_df(sql, db_config, columns=None):
    '''
    从mysql数据库中查询数据并存为dataframe
    Args:
        sql: string, sql查询语句
        db_config: dict, 数据库配置
        columns: list, 列名，默认为None
    Return:
        df: DataFrame
    '''
    conn = connect_mysql(db_config)
    try:
        query_result = conn.execute(sql.split(';')[0])
    except Exception as error:
        raise error

    df = pd.DataFrame(query_result.fetchall(), columns=columns)
    conn.close()

    return df
