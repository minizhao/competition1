任务1：学者画像数据集

 

画像数据统计信息：本数据集由 Aminer 提供，共包含 15000 位学者的个人画像数据（收集时间：2017 年7月10日）。

学者画像数据集共包含以下两个文件：

1、training.txt：训练数据（包含6000个学者）

不同学者的兴趣标签数据之间用空行分隔，具体格式如下
字段 	含义
#id 	学者的id
#name 	学者的名字
#org 	学者所在的机构名
#search_results_file 	搜索结果的文件名
#homepage 	学者的个人主页
#pic 	学者主页上识别的照片（链接）
#email 	学者主页上识别的email（也有可能是链接）
#gender 	学者的性别（male/female，以第一个字符为准）
#position 	学者的职称/职位列表（xxx;xxx;...）
#location 	学者所在机构的国家（country）

详细说明请参见：

任务一详细规则

2、validation.txt：验证数据（包含2434个学者）

不同学者的画像数据之间用空行分隔，具体格式如下
字段 	含义
#id 	学者的id
#name 	学者的名字
#org 	学者所在的机构名
#search_results_file 	搜索结果的文件名
