### 知识图谱构建探究

> github关键词：知识图谱，按星标数筛选了100星以上的项目作知识图谱构建方法调查
>
> 工具发现：Mongo、Deepdive
>
> [OpenKG ](http://www.openkg.cn/home)：（网站）包括知识图谱数据集、工具等
>
> [DeepKE](https://github.com/zjunlp/DeepKE) ：中文知识图谱抽取工具：实体抽取、关系抽取、属性抽取。
>
> [Deepdive](http://www.openkg.cn/dataset/cn-deepdive#)：deepdive是由斯坦福大学InfoLab实验室开发的一个开源知识抽取系统。它通过弱监督学习，从非结构化的文本中抽取结构化的关系数据 。本项目修改了自然语言处理的model包，使它支持中文，并提供中文tutorial。
>
> 关系抽取**远程监督**（用于标注）(远程监督也可以用于实体抽取）：见paper

------

[TOC]

#### 一、funNLP

> https://github.com/fighting41love/funNLP

内容包括：词库、工具包、学习资料

#### 二、从无到有搭建以疾病为中心的医药领域知识图谱&自动问答任务

> https://github.com/liuhuanyong/QASystemOnMedicalKG

本项目立足医药领域，以垂直型医药网站为数据来源，以疾病为核心，构建起一个包含**7类**规模为4.4万的**知识实体**，**11类**规模约30万**实体关系**的知识图谱。

 本项目将包括以下两部分的内容：

- 基于垂直网站数据的医药知识图谱构建

- 基于医药知识图谱的自动问答

#### 三、农业知识图谱

> https://github.com/qq547276542/Agriculture_KnowledgeGraph

该DEMO包括：实体识别、实体查询、关系查询、农业知识概览、农业知识问答、农业智能决策

#### 四、awesome-knowledge-graph

> https://github.com/husthuke/awesome-knowledge-graph

整理知识图谱相关学习资料，提供系统化的知识图谱学习路径。

#### 五、小型金融知识图谱

> https://github.com/jm199504/Financial-Knowledge-Graphs

#### 六、证券知识图谱

> https://github.com/lemonhu/stock-knowledge-graph

- 从网页中抽取董事会的信息
- 获取股票行业和概念的信息
- 设计知识图谱
  - 创建实体
  - 创建关系
- 创建可以导入Neo4j的csv文件
- 利用csv文件生成数据库
- 基于构建好的知识图谱，通过编写Cypher语句回答问题

#### 七、电影领域问

> https://github.com/SimmerChan/KG-demo-for-movie

#### 八、中文人物关系知识图谱

> https://github.com/liuhuanyong/PersonRelationKnowledgeGraph

#### 九、开源web知识图谱

> https://github.com/lixiang0/WEB_KG

- 爬取百度百科中文页面
- 解析三元组和网页内容
- 构建中文知识图谱
- 构建百科bot

#### 十、金融领域知识图谱

> https://github.com/Skyellbin/neo4j-python-pandas-py2neo-v3

利用pandas将excel中数据抽取，以三元组形式加载到neo4j数据库汇总构建相关知识图谱

#### 十一、刘焕勇相关知识图谱项目

> https://liuhuanyong.github.io/

- 技术思考
- 开源项目
  - 常识推理
  - 系统平台
  - 知识问答
  - 知识图谱
  - 语言资源
  - 语言工具
  - 信息抽取
  - 信息采集
  - 文本生成
  - 文本计算
  - 事理抽取
  - 情感计算

#### 十二、基于深度学习的开源中文知识图谱抽取框架

> https://github.com/zjunlp/DeepKE/blob/main/README_CN.md

DeepKE 是一个支持低资源、长篇章的知识抽取工具，可以基于PyTorch实现命名实体识别、关系抽取和属性抽取功能。同时为初学者提供了详尽的[文档](https://zjunlp.github.io/DeepKE/)，[Google Colab教程](https://colab.research.google.com/drive/1vS8YJhJltzw3hpJczPt24O0Azcs3ZpRi?usp=sharing)和[在线演示](http://deepke.zjukg.cn/CN/index.html)。

#### 十三、农业知识图谱

> https://github.com/CrisJk/Agriculture-KnowledgeGraph-Data

**发现标注关系的工具 MongoDB**

包括爬取Wikidata数据的爬虫、爬取复旦知识工场数据的爬虫(由于知识工场限制爬取，这部分暂时不好用)、提取所有中文维基页面的脚本以及将Wikidata三元组数据对齐到中文维基页面语句的脚本。

#### 十四、豆瓣知识图谱

> https://github.com/weizhixiaoyi/DouBan-KGQA

本项目构建了完整的基于豆瓣电影、书籍的知识图谱问答系统，包括从豆瓣原始数据的爬取、豆瓣原始数据转换得到RDF数据、三元组数据的存储与检索、问句理解和答案推理**、**微信公众号部署**等环节

#### 十五、豆瓣图书知识图谱

> https://github.com/mattzheng/DouBanRecommend

#### 十六、985高校知识图谱

> https://github.com/s-top/Baike-KnowledgeGraph

#### 十七、海贼王知识图谱

> https://github.com/mrbulb/ONEPIECE-KG

包括数据采集、知识存储、知识抽取、知识计算、知识应用。

#### 十八、从零构建百度百科/电影知识图谱

> https://github.com/Pelhans/Z_knowledge_graph

##### 1 数据获取

###### 1.1	半结构化数据获取

半结构化数据从百度百科和互动百科获取，采用scrapy框架，目前电影领域和通用领域两类。

- 通用领域百科数据：百度百科词条4,190,390条，互动百科词条3,677,150条。

> http://pelhans.com/2019/01/04/kg_from_0_note7/

- 电影领域: 百度百科包含电影22219部，演员13967人，互动百科包含电影13866部，演员5931 人

> http://pelhans.com/2018/09/01/kg_from_0_note1/

###### 1.2 非结构化数据（获取文本）

非结构化数据主要来源为微信公众号、虎嗅网新闻和百科内的非结构化文本。

微信公众号爬虫获取公众号发布文章的标题、发布时间、公众号名字、文章内容、文章引用来源；虎嗅网爬虫 获取虎嗅网新闻的标题、简述、作者、发布时间、新闻内容

##### 2 非结构化文本的知识抽取

###### 2.1 基于Deepdive的知识抽取

Deepdive是由斯坦福大学InfoLab实验室开发的一个开源知识抽取系统。它通过弱监督学习，从非结构化的文本中抽取结构化的关系数 据 。本次实战基于OpenKG上的[支持中文的deepdive：斯坦福大学的开源知识抽取工具（三元组抽取）](http://www.openkg.cn/ dataset/cn-deepdive)，我们基于此，抽取电影领域的演员-电影关系。

> 详细介绍见：http://pelhans.com/2018/10/10/kg_from_0_note5/

###### 2.2 神经网络关系抽取

利用自己的百科类图谱，构建远程监督数据集，并在OpenNRE上运行。最终生成的数据集包含关系事实18226，无关系(NA)实体对336 693，总计实体对354 919，用到了462个关系(包含NA)。

> 详细介绍见：http://pelhans.com/2019/01/04/kg_from_0_note9/

##### 3 结构化数据到RDF（三元组）

结构化数据到RDF由两种主要方式，一个是通过[direct mapping](https://www.w3.org/TR/rdb-direct-mapping/)，另一个通过[R2RML](https://www.w3.org/TR/r2rml/#acknowledgements)语言这种，基于R2RML语言的方式更为灵活，定制性强。对于R2RML有一些好用的工具，此处我们使用d2rq工具，它基于R2RML-KIT。

> http://pelhans.com/2019/02/11/kg_from_0_note10/

##### 4 存储数据Neo4j

图数据库是基于图论实现的一种新型NoSQL数据库。它的数据数据存储结构和数据的查询方式都是以图论为基础的。图论中图的节本元素为节点和边，对应于图数据库中的节点和关系。我们将上面获得的数据存到 Neo4j中。

> 百科类图谱：http://pelhans.com/2019/01/04/kg_from_0_note8/
>
> 电影知识图谱：http://pelhans.com/2018/11/06/kg_neo4j_cypher/

##### 5 语义检索（知识问答）



