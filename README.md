## 基本信息
#### 姓名：邱景诚
#### 毕业院校：重庆大学
#### 学历：硕士
#### 邮箱：15775938635@163.com

## 工作经历
### 信也科技（2017.7 - 至今）
#### 数据挖掘专家
- 负责主营业务复借用户贷款审批模型的迭代、上线部署以及维护工作；
- 为贷后资产管理部门从0到1搭建起整个模型体系；
- 负责团队建模工具的开发和维护。

## 代码索引
### 通用工具
包含日常工作中使用的工具函数，例如自动编辑excel、发邮件、与mysql、hive交互等。
- **autoexcel**：自动编辑excel，支持将DataFrame和图片插入excel
- **general_tools**：一些日常使用的小工具
- **hdfs_reader**：与HDFS交互的API，从HDFS中下载文件并读取成DataFrame
- **hive_api**：与Hive交互的API
- **mysql_api**： 与Mysql交互的API
- **pandas_sql_func**： 在DataFrame上实现一些hive函数相同的功能
- **send_email**：邮件发送工具, 可自动生成html表格，在正文中插入图片和附件
- **variable_dictionary**：根据SQL代码中的注释自动生成数据字典

### 建模工具
建模流程中使用的工具函数，包含变量EDA、数据预处理、模型训练、模型验证、模型监控等；
- **data_overview**：数据探索性分析工具包
- **int_encoder**：正整数编码工具
- **preprocess**： 数据预处理工具包
- **woe_tools**：变量WOE计算工具包
- **model_building**：建模工具包
- **model_validation**： 模型效果验证工具包
- **model_monitor**：模型监控工具包

### 算法研究
实现论文中的算法
- **tradaboost**：TradaBoost迁移学习算法，相关论文《Boosting for transfer learning》
- **ftrl**：谷歌的FTRL在线学习算法，相关论文《Ad Click Prediction- a View from the Trenches》
- **gbdt**：梯度提升树算法，相关论文《Greedy function approximation: A gradient boosting machine》

