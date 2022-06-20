## A Walk-based Model on Entity Graphs for Relation Extraction

#### 1 Introduction

**论文动机**

一句话中可能存在多个关系-实体对，RE模型需要考虑这些实体对之间的依赖关系，以得到更准确的模型。作者将一句话中的多个三元组建模为一个图结构，图的节点对应实体，图的边对应关系，用l-length游走路径来联系不同的实体对。

**论文贡献**

- We propose a graph walk based neural model that considers multiple entity pairs in relation extraction from a sentence.
- We propose an iterative algorithm to form a single representation for up-to l-length walks between the entities of a pair.
- We show that our model performs comparably to the state-of-the-art without the use of external syntactic tools.

#### 2 Model

##### 2.1 Embedding Layer

实体的每个token的embedding由三部分组成$w=[n_w,n_t,n_p]$.分别表示word embedding，semantic entity types，relative position to target entities。而非实体的token则由前两部分构成，没有相对位置的嵌入。其中，$n_p$表示，一个实体距离另一个实体的距离。

##### 2.2 Bidirectional LSTM Layer

$e_t=[\overrightarrow{h_t};\overleftarrow{h_t}]$

作者还阐述了为什么不在BiLSTM层编码实体对之间的关系：（1）由于BiLSTM以序列为单位展开相关计算，而不是以实体对为单位，因此计算成本降低；（2）可以在句子对之间共享序列层，使得模型能间接了解句子中的隐藏依赖关系。

##### 2.3 Edge Representation Layer

作者将BiLSTM的输出做两部分的划分，并做不同的处理：（1）实体对；（2）文本中除了实体对的token。

**实体对 Entity Pair**

如果一个实体包括了N个单词，则将实体的向量表示为：$e=\frac{1}{|I|}\sum_{i\in I}e_i$，即做一个取平均运算。

对每一个实体，向量表达为:$v_i=[e_i;t_i;p_{ij}]$。其中，$e_i$为取平均的结果，$t_i$为实体类型的嵌入，$p_{ij}$为与另一个实体的相对位置关系。

**Context of the Entity Pair**

对每一个非实体的token，向量表达为$v_{ijz}=[e_z;t_z;p_{zi};p_{zj}]$。其中，$e_z$为BiLSTM的输出，$t_z$为实体类型的嵌入，$p_{zi}$为距离头实体的相对距离，$p_{zj}$为距离尾实体的相对距离。

---

由上，一句话可以被一个三维的矩阵$C$表示出来。行列代表实体对，深度代表与实体对对应的上下文。

利用Attention机制计算上下文的权重以及加权平均。
$$
u=q^Ttanh(C_{ij})\\
\alpha=softmax(u)\\
c_{ij}=C_{ij}\alpha^T
$$
计算出的$c_{ij}$与target pair拼接，通过全连接层得到输出向量。这对应one-length walk。

##### 2.4 Walk Aggregation Layer

> Our main aim is to support the relation between an entity pair by using chains of intermediate relations between the pair entities. Thus, the goal of this layer is to generate a single representation for a finite number of different lengths walks between two target entities.

该层是为两个目标实体之间有限数量的不同长度的行走生成单个表示。实现方法：将一句句子表示为有向图，其中实体对应节点，关系对应有向边。