# coding: utf-8
# Author: Jingcheng Qiu

"""
与HDFS交互的API，从HDFS中下载文件并读取成DataFrame
"""


import os
import hdfs
from datetime import datetime
from threading import Thread
from pandas import read_csv, concat
from pyarrow.parquet import read_table


class Threading(Thread):
    """
    多线程执行函数，并保存函数的返回值
    """
    def __init__(self, func, args=()):
        """
        Parameters
        ----------
        func: function
            执行函数
        args: tuple or dict
            函数参数
        """
        super(Threading, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        if isinstance(self.args, tuple):
            self.result = self.func(*self.args)
        elif isinstance(self.args, dict):
            self.result = self.func(**self.args)
        else:
            raise ValueError('the type of "args" can only be tuple or dict')

    def get_result(self):
        Thread.join(self)
        try:
            return self.result
        except Exception:
            return None


class HdfsReader(object):
    """
    从HDFS上下载文件再读取成DataFrame
    """
    def __init__(self, hdfs_client, impala_client=None):
        """
        Parameters
        ----------
        hdfs_client: hdfs.InsecureClient object
            HDFS代理
        impala_client: impala.dbapi.connect object, default None
            impala代理, 若不知道HDFS文件的具体路径，需指定该参数，以获取表文件存放的HDFS路径
        """
        self.client = hdfs_client

    def download_file_from_hdfs(self, hdfs_path, save_path=None):
        """
        从HDFS中下载文件
        Parameters
        ----------
        hdfs_path: str,
            HDFS路径
        save_path: str, default None
            文件在本地存放的路径，若不填则会在当前工作目录下创建一个hdfs_files的文件夹存放文件
        """
        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'hdfs_files')    # 存放parquet文件的临时目录

        # 建立存放文件的文件夹，若文件夹已存在则清空其中的文件
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            for f in os.listdir(save_path):
                os.remove(os.path.join(save_path, f))

        # 下载文件
        if self.client.status(hdfs_path)['type'] == 'FILE':
            self.client.download(hdfs_path, os.path.join(save_path, os.path.basename(hdfs_path)))
        else:
            start_time = datetime.now()
            cnt = 0
            for f in self.client.list(hdfs_path):
                if self.client.status(os.path.join(hdfs_path, f))['type'] == 'FILE':
                    self.client.download(os.path.join(hdfs_path, f), os.path.join(save_path, f))
                    cnt += 1
                    print('\r', 'Successfully Download {0} files'.format(cnt), end='', flush=True)
            print('Time cost: {1}'.format(cnt, datetime.now() - start_time))

        return 'Download completed'

    @staticmethod
    def read_table_from_file(file_path, file_type='parquet', columns=None):
        """
        读取文件生成DataFrame
        Parameters
        ----------
        file_path: str
            文件在本地存放的路径
        file_type: str, default 'parquet', options ['parquet', 'textfile']
            文件的格式
        columns: list, default None
            DataFrame列名

        Returns
        -------
        df: DataFrame
        """
        # 读取文件并转成DataFrame
        if file_type == 'parquet':
            df = read_table(file_path, use_legacy_dataset=True).to_pandas()
        elif file_type == 'textfile':
            threads = []
            for f in os.listdir(file_path):
                args = {'filepath_or_buffer': os.path.join(file_path, f),
                        'sep': '\001',
                        'header': None,
                        'names': columns,
                        'na_values': '\\N'}
                task = Threading(func=read_csv, args=args)
                task.start()
                threads.append(task)
            for task in threads:
                task.join()
            container = [t.get_result() for t in threads]
            df = concat(container, axis=0, ignore_index=True)
            del container, threads
        else:
            raise ValueError('file_type can only be "parquet" or "textfile"')

        return df
