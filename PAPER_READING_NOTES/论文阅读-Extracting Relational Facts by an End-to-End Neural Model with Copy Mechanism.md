## Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism

>  2018-ACL
> Zeng X, Zeng D, He S, et al. 
> "Extracting relational facts by an end-to-end neural model with copy mechanism." 

作者提出关系三元组在同一句子中可能存在重叠，并将其分为三类情况：

1. Normal：没有实体重叠
2. EntityPairOverlap（EPO）：实体对重叠
3. SingleEntityOverlap（SEO)：单个实体重叠

![image-20220613100829734](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613100829734.png)

现有大多数方法只对Normal子类进行研究，忽略了一个实体可能存在多关系的情况。这篇文章指向的问题是**关系三元组重叠**的情况。

---

#### 1  Introduction

为了解决这个问题，作者提出基于Seq2Seq和Copy Mechanism的模型，用于从句子中提取关系三元组。模型主要包括Encoder-Decoder两部分。

- 编码器将源句子转换为固定长度的语义向量
- 解码器读入向量，生成三元组
  - 解码器生成关系
  - 复制机制，从源语句复制头实体
  - 复制机制，从源语句复制尾实体

当一个实体参与不同三元组时，它可以被复制多次，因此模型可以处理重叠问题。

文章还提出解码过程的两个不同策略：OneDecoder & MultiDecoder

**Contributions**

- Propose an end2end neural model based on sequence-to-sequence learning with copy mechanism to extract relational facts from sentences, where the entities and relations could be jointly extracted.
- Our model could consider the relational triplet overlap problem through copy mechanism. In our knowledge, the relational triplet overlap problem has never been addressed before.
- We conduct experiments on two public datasets. Experimental results show that we outperforms the state-of-the-arts with 39.8% and 31.1% improvements respectively.

---

#### 2 Model

##### 2.1 OneDecoder Model

###### 2.1.1 Encoder

Bi-directional RNN将文本$$s=[w_1,..,w_n]$$编码为固定长度的向量$$O^E=[o_1^E,...,o_n^E],o_t^E=[\overrightarrow{o_t^E}；\overleftarrow{o_{n-t+1}^E}]$$，同时RNN隐藏层输出$s=[\overrightarrow{h_n^E};\overleftarrow{h_n^E}]$。

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613103622295.png" alt="image-20220613103622295" style="zoom: 67%;" />

###### 2.1.2 Decoder

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613105025734.png" alt="image-20220613105025734" style="zoom: 67%;" />

Figure3中，$g$为decoder function，$h_0^D$由编码器隐藏层输出$s$初始化。$c_t$为attention vector，$v_t$为t-1步骤中复制过来的实体或预测的关系的embedding。
$$
o_t^D,h_t^D=g(u_t,h_{t-1}^D)\\
u_t=[v_t;c_t]\cdot{W^u}
$$
**Attention Vector $$c_t$$**
$$
c_t=\sum_{i=1}^n\alpha_i\times{o_i^E}\\
\alpha=softmax(\beta)\\
\beta_i=selu([h_{t-1}^D;o_i^E]\cdot{w^c})
$$
$o_i^E$是Encoder的$i$步骤输出；$w^c$是weight vector.

在三个时间步预测一个关系，另外两个时间步分别复制头实体和尾实体。

**Predict Relation**

假设有m个关系，用全连接层计算置信向量$q^r=[q_1^r,...,q_m^r]$:
$$
q^r=selu(o_t^D\cdot{W^r}+b^r)
$$
同时，预测关系时，可能关系三元组是NA-relation的，这也应该考虑进来。
$$
q^{NA}=selu(o_t^D\cdot{W^{NA}}+b^{NA})
$$
将$q^r$和$q^{NA}$拼接，通过softmax得到$p^r$
$$
p^r=softmax([q^r;q^{NA}])
$$
生成$p^r$后，选择概率最大的关系作为预测关系，并将其embedding作为下一时间步的输入$v^{t+1}$

**Copy the First Entity**

复制实体，首先计算$q^e=[q_1^e,...,q_n^e]$：
$$
q_i^e=selu([o_t^D;o_i^E]\cdot{w^e})
$$
与关系抽取类似，允许实体为NA。
$$
p^e=softmax([q^e;q^{NA}])
$$
选择最大概率的实体，并将其embedding作为下一时间步的输入$v_{t+1}$

**Copy the Second Entity**
复制尾实体和复制头实体几乎相同，唯一区别的是复制尾实体时，尾实体不能与头实体相同。用掩码向量M来实现
$$
M_i=\begin{cases} 1,i\neq k\\0,i =k \end{cases}\\
p^e=softmax([M\otimes q^e;q^{NA}])
$$

##### 2.2 MultiDecoder Model

一个解码器对应一个三元组。解码器之间的衔接：
$$
\hat{h}_{t-1}^{D_i}= \begin{cases} s, i=1\\ \frac{1}{2}(s+h_{t-1}^{D_{i-1}}),i>1 \end{cases}
$$

---

#### 3 Experiments

**Datasets**：NYT、WebNLG

**Results** 
baseline：《Joint extraction of entities and relations based on a novel tagging scheme.》

![image-20220613115623375](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613115623375.png)

![image-20220613115747475](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613115747475.png)

![image-20220613115800841](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613115800841.png)

![image-20220613115814818](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220613115814818.png)



#### 4 后话

《CopyMTL: Copy mechanism for joint extraction of entities and relations with multi-task learning》这篇文章对CopyRE进行了详细的分析，发现如下两个缺陷：

> 我们证明CopyRE实际上使用相同的分布来建模头实体h和尾实体t，选择最高概率作为h，在掩盖了最高概率后，选择次高概率作为t，因此没有此掩码，它无法区分h和t。以这种方式建模h和t分布可能会导致各种问题，模型不仅在不同的h和t下非常弱，但在预测t时，也无法获得关于h的信息。 
>
> 其次，CopyRE无法提取具有多个标记的实体。基于Copy Mechanism的解码器总是指向任何实体的最后一个标记，这限制了模型的适用性。例如，当实体有两个标记时，如Steven Jobs，CopyRE仅预测“Jobs”，而不是整个实体“Steven  Jobs”。在真实的word场景中，多标记实体很常见，因此这会极大地影响模型性能。

这两点也是CopyMTL一文着重克服的问题

