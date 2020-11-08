# encoding: utf-8
# Author: Jingcheng Qiu

"""
发邮件工具, 可自动生成html表格，在正文中插入图片和附件
"""


import os
import smtplib
from email import encoders, MIMEMultipart
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from jinja2 import Template


def get_content(file_path, limit_line=0):
    """
    将文本文件转成html格式的字符串
    Parameters
    ----------
    file_path:: str
        文件的绝对路径
    limit_line: int, default 0
        限制显示的行数(从最后往前数)，0表示显示文件的全部内容
    
    Returns
    -------
    html: str
        html格式的字符串    
    """
    html = ''
    with open(file_path, 'r') as f:
        content = f.readlines()
        for line in content[-limit_line:]:
            html = html + line + '<br />'

    return html


def create_image_html(images):
    """
    生成嵌入图片的html
    Parameters
    ----------
    images: List
        嵌入图片列表, eg.
        [{'title': 'graph1', 'image': '/home/test/a.png', 'shape': (1400, 1100)},
         {'title': 'graph2', 'image': '/home/test/b.png', 'shape': (1400, 1100)},
        ]
        注意: 列表里面的顺序就是图片显示的顺序
        shape 是可选项, 表示图片大小, 默认 (1400, 1100)

    Returns
    -------
    html: str
        html格式的字符串
    """
    range_images = range(len(images))
    shapes = [i.get('shape', (1400, 1100)) for i in images]
    html_template = '''
    {% for i in range_images %}
        <p>
        <b>{{ images[i]["title"] }}</b><br>
        <img src="cid:image{{ i+1 }}" height="{{ shapes[i][0] }}" width="{{ shapes[i][1] }}"">
        </p>
    {% endfor %}
    '''
    template = Template(html_template.replace('\n', ''))
    html = template.render(range_images=range_images, images=images, shapes=shapes)
    
    return html


def create_table_html(tables):
    """
    生成表格的html
    Parameters
    ----------
    tables: dict
        表格内容, eg.
        [
         {'title': 'table1',
          'table': [{'t1': 1, 't2': 2, 't3': 'a'},
                    {'t1': 10, 't2': 12, 't3': 'bb'},
                   ]
         },
         {'title': 'table2',
          'table': [{'t1': 1, 't2': 2, 't3': 'a'},
                    {'t1': 10, 't2': 12, 't3': 'b'},
                   ]
         }
        ]
        注意: 列表里面的顺序就是表格显示的顺序

    Returns
    -------
    html: str
        html格式的字符串
    """
    new_tables = []
    for t in tables:
        title = t['title']
        table = t['table']
        header = set()
        for col in table:
            header = header.union(col.keys())
        new_tables.append({'title': title, 'table': table, 'header': header})

    html_template = '''
    {% for table in new_tables %}
        <p>
        <b>{{ table["title"] }}</b><br>
        <table border="1">
            <tr>
            {% for h in table["header"] %}
                <th>{{ h }}</th>
            {% endfor %}
            </tr>
            {% for t in table["table"] %}
                <tr>
                {% for h in table["header"] %}
                    <td>{{ t.get(h) }}</td>
                {% endfor %}
                </tr>
            {% endfor %}
        </table>
        </p>
    {% endfor %}
    '''
    template = Template(html_template.replace('\n', ''))
    html = template.render(new_tables=new_tables)
    
    return html


def transcode_addr(s):
    """
    邮件地址转码
    """
    name, addr = parseaddr(s)
    
    return formataddr((\
        Header(name, 'utf-8').encode(),\
        addr.encode('utf-8') if isinstance(addr, unicode) else addr))


def send_email(host, port, sender, to, cc=[], subject='', content='', file_path=[], image_path=[], user=None,
               password=None, ssl=False):
    """
    Parameters
    ----------
    host: str
        邮件服务器
    port: str
        邮件服务器端口
    to: list
        收件人列表
    cc: list, default []
        抄送列表
    content: str, default ''
        文字内容
    subject: str, default ''
        邮件主题
    file_path: str or list, default []
        附件路径
    image_path: str or list, default []
        图片路径
    user: str, default None
        邮箱账号
    password: str, default None
        登录用户密码
    ssl: bool, default False
        是否通过SSL的方式连接邮箱
    """
    # 初始化邮件，定义邮件的格式
    msgRoot = MIMEMultipart('related') ##采用related定义内嵌资源的邮件体

    msgRoot['From'] = transcode_addr(sender)
    msgRoot['To'] = ';'.join(to)
    if len(cc) > 0:
        msgRoot['Cc'] = ';'.join(cc)
    msgRoot['Subject'] = Header(subject, 'utf-8')
    msgText = MIMEText(content, 'html', 'utf-8')
    msgRoot.attach(msgText)
    
    # 插入图片
    if image_path is not None:
        for i, imgpath in enumerate(image_path):
            img = open(imgpath, 'rb')
            msgImage = MIMEImage(img.read())
            img.close()
            msgImage.add_header('Content-ID', '<image{0}>'.format(i+1))
            msgRoot.attach(msgImage)

    # 添加附件
    if type(file_path) == str:
        file_path = [file_path]
    if len(file_path) > 0:
        for fp in file_path: 
            attachment = MIMEBase('application', "octet-stream")   
            attachment.set_payload(open(fp, "rb").read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(fp)))
            msgRoot.attach(attachment)

    # 发送邮件
    server = smtplib.SMTP()
    server = smtplib.SMTP_SSL(host, port) if ssl else smtplib.SMTP(host, port)
    if user:
        server.login(user, password)
    server.sendmail(sender, to+cc, msg=msgRoot.as_string())
    server.close()
    
    return True
