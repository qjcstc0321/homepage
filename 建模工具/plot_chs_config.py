# coding: utf-8

"""
用于matlibplot绘图显示中文字符, 需要代码路径下存在中文字体文件'simsun.ttc'
"""


import os
import sys
from matplotlib.font_manager import FontProperties


if sys.version_info.major == 2:
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.setdefaultencoding('utf-8')
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde

try:
    path_abs = os.path.dirname(os.path.realpath(__file__))
    zhfont = FontProperties(fname=os.path.join(path_abs, 'simsun.ttc'))
except Exception:
    print('cannot find zhfont')
    zhfont = FontProperties()
