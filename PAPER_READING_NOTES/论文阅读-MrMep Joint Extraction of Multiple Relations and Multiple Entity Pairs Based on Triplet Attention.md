## MrMep: Joint Extraction of Multiple Relations and Multiple Entity Pairs Based on Triplet Attention

> 2019-CoNLL
> Jiayu Chen, Caixia Yuan, Xiaojie Wang, etc.
> Center of Intelligence Science and Technology School of Computer Science
> Beijing University of Posts and Telecommunications（北邮）

问题概要：实体关系联合抽取、关系重叠、多词实体

#### 1 Introduction

联合抽取任务的理想目标是抽取文本中所有可能的关系类型，并提取每个目标关系类型的所有候选实体对，同时考虑到三元组之间的复杂重叠。

面对这一任务，《Joint extraction of entities and relations based on a novel tagging scheme》提出一种标记机制，通过将关系类型和实体位置信息注入标记中，将关系事实提取转化为序列标记任务。在这种paradigm中，Tagging model为每个单词指定了一个唯一的标签，这使得模型无法提取具有重叠实体甚至重叠实体对的三元组。

《CopyRE: Extracting relational facts by an end-to-end neural model with copy mechanism》使用Seq2Seq+Copy Mechanism考虑到实体重叠的情况，但无法处理一个实体对应多个单词的情况。

《A hierarchical framework for relation extraction with reinforcement learning》应用带有强化学习的分层框架，将三元组提取分解为用于关系检测的high-level task和用于实体提取的low-level task（此后称为HRL）。 

这篇文章提出的MrMep(a novel approach for jointly extracting Multiple Relations and Multiple Entity Pairs).它利用了Triplet Attention来刻画实体-关系间的暗含关系，然后对于每个目标关系，它使用指针网络的变体以顺序方式生成所有头实体和尾实体的边界（起始/结束位置），由此模型生成所有可能的实体。

**Contributions**

- We propose MrMep, a novel neural method which firstly extracts all possible relations and then extracts all possible entity pairs for each target relation, while the two procedures are packed together into a joint model and are trained in a joint manner.
- MrMep uses a triplet attention to strengthen the connections among relation and entity pairs, and is computationally efficient for sophisticated overlapped triplets even with lightweight network architecture.
- Through extensive experiments on three benchmark datasets, we demonstrate MrMep’s effectiveness over the most competitive state-of-the-art approach by 7.8%, 7.0% and 9.9% improvement respectively in F1 scores.

**与CopyRE、HRL的区别**

MrMep利用traplet attention的解码器可以对关系和实体对之间的交互进行建模。通过让解码器只预测实体对，减轻了它重复生成关系类型的负担。

#### 2 Model

##### 2.1 Overview

文本段落$S=[w_1,...,w_n]$，$R$是预定义关系类型的集合。任务为预测可能的关系组序列$<e_i,r_{ij},e_j>$。
MrMep总体架构由三个主要部分组成：Encoder, Multiple Relation Classifiers, Vairable-length Entity Pair Predictor。
<u>Encoder</u>: 使用LSTM对源文本进行预处理并提取序列级特征
<u>Multiple Relation Classifiers</u>: 预测S中可能存在的关系
<u>Variable-length Entity Pair Predictor</u>: 依次为每一种可能的关系类型生成所有可能的实体对

![image-20220614203045510](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614203045510.png)

##### 2.2 Encoder

给定文本段落$S=[w_1,...,w_n]$，获得对应的文本embedding，随后使用LSTM学习每个单词$w_i$的token representation $X_n$.

##### 2.3 Multiple Relation Classifiers

在CNN卷积部分，文章采用《Convolutional Neural Networks for Sentence Classification》一文中提出的卷积结构（max-over-time pooling）。该CNN模块输出text embedding$Q$与原token representation $X_n$拼接，得到fused vector $H=Concat(Q,X_n)$.

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614203621931.png" alt="image-20220614203621931" style="zoom: 67%;" />

接着经过一层线性层，一层线性层和一层softmax层。值得注意的是，最终softmax输出为概率分布，表示该文本是否包含第j个关系。第一层线性层输出为第j个关系的嵌入 relation embedding $R_j$。

##### 2.4 Variable-length Entity Pair Predictor

受指针网络启发(《Pointer networks》)，通过识别文本中单词的起始和结束位置索引来确定实体。figure2右侧，每个实体有两个对应的起始索引和结束索引，每两个实体按顺序形成一个实体对。

**Multi-head Attention**

多头自注意力机制，和原论文《Attention is all you need》提出的一样。输入为$X_n$，经过（1），输出为text representation $P$。
$$
Q=XW_j^Q,K=XW_j^K,V=XW_j^V\\
head_j=softmax(\frac{QK^T}{\sqrt{d_k}})V,\\
P=Concat(head_1,...,head_h)W^O
$$
**Triplet Attention**

在文本每个位置，注意力机制用于获得表示token和target relation之间匹配程度的加权值。由于其目的是提取目标关系的候选实体对，因此作者将其称之为Triplet Attention.这本质上是将关系类型的匹配聚合到文本的每个标记，并使用聚合的匹配结果进行实体预测。

假设j关系为关系分类器的预测关系，作者研究了两种不同的Triplet Attention实现模式：Paralleled Mode & Layered Mode.

**Paralleled Mode**
$$
a_t^i=W^atanh(W^r\circ R_j+W^d\circ d_{t-1} + W^P\circ P_i)
$$
$a_t^i$: attention weight of i-th word in the text
$W^a,W^r,W^d,W^p$: learnable parameters
$R_j$: j-th relation embedding
$d_{t-1}$: hidden state of LSTM decoder at time step t-1
$P_i$: i-th word of text representation.
$\circ$：element-wise multiplication operation

随后计算 attention distribution on text $\alpha_t$。将t时间步最高概率的$\alpha_t^i$作为输出$O_t$.
$$
\alpha_t=softmax(a_t)
$$
随后更新Decoder-LSTM的相关参数
$$
c_t=\sum_{i=1}^n\alpha_t^i\cdot P_i\\
d_t=LSTM_{decoder}(c_t,d_{t-1})
$$
$c_t$: context vector
$d_{t-1}$: hidden state of LSTM decoder at time step t-1

**Layered Mode**
$$
\beta_t=tanh(W^{r’}\circ R_j + W^{d'} \circ d_{t-1})\\
a_i^t=W^{a'}tanh(W^\beta \circ \beta_t + W^{p'} \circ P_i)
$$
$\beta_t$: 中间变量
$W^\beta,W^{p'},W^{a'}$: learnable parameters

#### 3 Experiments

**Datasets**: NYT, WebNLG, SKE

**Results**:

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614213818479.png" alt="image-20220614213818479" style="zoom:67%;" />

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614213823826.png" alt="image-20220614213823826" style="zoom:67%;" />

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614213848059.png" alt="image-20220614213848059" style="zoom:67%;" />

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220614213900013.png" alt="image-20220614213900013" style="zoom: 50%;" />