## Transformer

>参考文章：
>
>https://zhuanlan.zhihu.com/p/48508221
>
>http://jalammar.github.io/illustrated-transformer/
>
>https://www.ylkz.life/deeplearning/p10770524/
>
>https://zhuanlan.zhihu.com/p/352233973
>
>https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#comment-4748247470
>
>https://medium.com/@_init_/how-self-attention-with-relative-position-representations-works-28173b8c245a

#### 前言

​		Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建，作者的实验是通过搭建编码器和解码器各6层，总共12层的Encoder-Decoder，并在机器翻译中取得了BLEU值得新高。

​		作者采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

- 时间片t的计算依赖t-1时刻的计算结果，限制了模型的并行能力。
- 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长的依赖现象，仍存在梯度消失/梯度爆炸的情况。

​        Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

> Transformer定义：Transformer is the first **transduction model** **relying entirely on self-attention** to compute representations of its input and output **without using sequence aligned RNNs or convolution**。

### Transformer详解

#### 1 Transformer主要框架

论文中transformer结构基于机器翻译，本质上，transformer是一个Encoder-Decoder（编码-解码）的结构，可以表示为如下结构：

<img src="http://jalammar.github.io/images/t/the_transformer_3.png" alt="img" style="zoom:67%;" />

<div align = "center">图1 transformer用于机器翻译</div>

<img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png" alt="img" style="zoom:80%;" />

<div align = "center">图2 transformer的Encoder-Decoder模型</div>

论文中，编码器由6个编码block组成，同样解码器是6个解码block组成。与所有的生成模型相同的是，编码器的输出会作为解码器的输入，如图3所示：

<img src="http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" alt="img" style="zoom: 50%;" />

<div align = "center">图3 transformer的Encoder-Decoder均由6个block堆叠而成</div>

每一个Encoder中，分别包含Self-Attention模块以及Feed Forward Neural Network模块。数据Q（query）、K（key）、V（value）通过Self-Attention模块，加权得到一个特征向量Z，公式可表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BAttention%7D%28Q%2CK%2CV%29%3D%5Ctext%7Bsoftmax%7D%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5Ctag1)

得到Z之后，它被送到下一个模块，即Feed Forward Neural Network， FFN。该模块由两层全连接层组成，第一层的激活函数为ReLU，第二层的激活函数为线性激活函数，可表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BFFN%7D%28Z%29+%3D+max%280%2C+ZW_1+%2Bb_1%29W_2+%2B+b_2+%5Ctag2)

<img src="http://jalammar.github.io/images/t/Transformer_encoder.png" alt="img" style="zoom: 67%;" />

<div align = "center">图4 Encoder结构</div>

Decoder的结构如图5所示，它与Encoder的不同之处在于Decoder多了一个Encoder-Decoder Attention，两个Attention分别用于计算输入和输出的权值：

- Self-Attention：当前翻译和已经翻译的前文之间的关系；

- Encoder-Decnoder Attention：当前翻译和编码的特征向量之间的关系。

<img src="http://jalammar.github.io/images/t/Transformer_decoder.png" alt="img" style="zoom: 80%;" />

<div align = "center">图5 Encoder-Decoder结构</div>

#### 2 输入编码

如图6所示，首先通过Word2Vec等词嵌入方法将输入语料转化成特征向量，论文中使用的词嵌入的维度为 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D%3D512) 。

<img src="http://jalammar.github.io/images/t/embeddings.png" alt="img" style="zoom:67%;" />

<div align = "center">图6 单词embedding</div>

在最底层的block中，x将直接作为Transformer的输入，而在其他层中，输入则是上一个block的输出。

<img src="http://jalammar.github.io/images/t/encoder_with_tensors_2.png" alt="img" style="zoom:50%;" />

<div align = "center">图7 Encoder第一层的tensor输入</div>

#### 3 Self-Attention

Attention机制由Bahdanau等人提出，核心内容是为输入向量的每个单词学习一个权重。当模型处理每个单词(输入序列中的每个位置)时，自我注意允许它查看输入序列中的其他位置，以寻找有助于更好地编码这个单词的线索。与RNN维护隐状态序列，将正在处理的单词与之前处理的单词连接起来 不同，自注意力机制“理解”其他单词以处理当前单词。

例如下面这句话，我们判断it指代的内容，

```text
The animal didn't cross the street because it was too tired
```

通过加权之后，可以得到类似图8的加权情况

![img](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

<div align = "center">图8 Self-Attention可视化视图</div>

在Self-Attention中，每个单词有3个不同的向量，它们分别是Query（Q）、Key（K）、Value（V），长度均为64。它们是通过嵌入向量X乘以三个不同的权值矩阵$$W^Q$$、$$W^K$$、$$W^V$$得到，其中三个矩阵的尺寸相同，均为512*64。

> Query，Key，Value的概念取自于信息检索系统，举个简单的搜索的例子来说。当你在某电商平台搜索某件商品（年轻女士冬季穿的红色薄款羽绒服）时，你在搜索引擎上输入的内容便是Query，然后搜索引擎根据Query为你匹配Key（例如商品的种类，颜色，描述等），然后根据Query和Key的相似度得到匹配的内容（Value)。

<img src="http://jalammar.github.io/images/t/transformer_self_attention_vectors.png" alt="img" style="zoom: 80%;" />

<div align = "center">图9 Q,K,V的计算示意图</div>

Self-Attention的计算过程可以分为7步：

1.将输入单词转化为embedding向量$$X$$；
2.根据嵌入向量$$X$$和权值矩阵$$W^Q$$、$$W^K$$、$$W^V$$得到$$q,k,v$$三个向量；
3.为每个向量计算一个score：$$score=q\cdot{k}$$。分数决定了当我们将一个单词编码到某个特定位置时，我们对输入句子的其他部分的关注程度;

> 可以引入不同的函数和计算机制,最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值,即如下公式
>
> $$Cosine相似性：Similarity(Query,Key ) = \frac{Query \cdot Key}{\parallel Query \parallel \cdot \parallel Key \parallel}$$

4.为了梯度的稳定，Transformer对score进行归一化，即除以$$\sqrt{d_k}$$（key向量的维度开方）。
5.对score施以softmax激活函数，使各值为正，且总和为1.这个softmax分数决定了每个单词在这个位置的表达量。显然，这个位置的单词将拥有最高的softmax分数，但有时关注与当前单词相关的另一个单词是有用的。
6.softmax点乘value值$$v$$，得到加权的每个输入向量的评分。
7.相加之后得到最终的输出结果$$z$$：$$z=\sum{v}$$。

<img src="http://jalammar.github.io/images/t/self-attention-output.png" alt="img" style="zoom: 80%;" />

<div align = "center">图10 Self-Attention计算过程</div>

实际实现中，为了更快地处理，这种计算往往是以矩阵形式进行的。

<img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation.png" alt="img" style="zoom:67%;" />

<div align = "center">图11 Q,K,V计算（矩阵形式)</div>

整个Self-Attention计算过程也可以用下图表示出来：

<img src="http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" alt="img" style="zoom:67%;" />

<div align = "center">图12 Self-Attention计算（矩阵形式)</div>

> 补充：注意力机制与自注意力机制的区别
>
> 传统的Attention机制在一般任务的Encoder-Decoder model中，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention机制发生在Target的元素Query和Source中的所有元素之间。简单的讲就是Attention机制中的权重的计算需要Target来参与的，即在Encoder-Decoder model中Attention权值的计算不仅需要Encoder中的隐状态而且还需要Decoder 中的隐状态。
>
> 而Self Attention顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。例如在Transformer中在计算权重参数时将文字向量转成对应的KQV，只需要在Source处进行对应的矩阵操作，用不到Target中的信息。

![img](http://jalammar.github.io/images/t/transformer_decoding_2.gif)

#### 4 多头注意力机制

多头注意力机制可以在如下两方面提升注意力层的性能：

- It expands the model's ability to focus on different positions.
- It gives the attention layer multiple "representation subspaces"。使用多头注意力机制，即拥有多个Query/Key/Value权重矩阵（每个均为随机初始化），经过训练，每个权重矩阵集被用来投射输入的嵌入到一个不同的表示子空间。

![img](http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

<div align = "center">图13 Multi-headed Attention</div>

Transformer使用8attention heads，此时不同权重矩阵集Q,K,V会生成八个不同的$$z$$矩阵.

<img src="http://jalammar.github.io/images/t/transformer_attention_heads_z.png" alt="img" style="zoom:67%;" />

<div align = "center">图14 8-heads attention</div>

由于每一个Encoder、Decoder中，Attention层后紧接着的Feed-forward层只需要一个矩阵（每个单词对应一个矩阵），所以需要将这8个$$z$$矩阵压缩成一个矩阵。transformer将这八个矩阵concat起来，然后乘以一个随模型训练的$$W^O$$矩阵。

<img src="http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png" alt="img" style="zoom: 50%;" />

<div align = "center">图15 压缩处理8 heads 产生的z输出矩阵</div>

综上所述，多头注意力机制可由下图（图16）表出：

![img](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

<div align = "center">图16 多头注意力机制总架构图</div>

~~~txt
The animal didn't cross the street because it was too tired
~~~

<img src="http://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png" alt="img"  />

<div align = "center">图17 多头注意力机制可视化</div>

#### 5 位置编码

截止目前为止，我们介绍的Transformer模型并没有捕捉顺序序列的能力。当一个序列输入进一个Self-Attention模块时，由于序列中所有的tokens是同时进入并被处理的，如果不提供位置信息，那么这个序列里的相同的token对Self-Attention模块来说就不会有语法和语义上的差别，它们会产生相同的输出。为了解决这个问题，论文中在编码词向量时引入了位置编码（Position Embedding）的特征。具体地说，位置编码会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词。

如何在输入序列中加入位置编码信息呢？
		一种方式是先把位置信息加入输入token的表达向量（比如通过预训练得到的词嵌入向量）里以后，再输入进Self-Attention模块。此时，位置向量的产生是在Self-Attention模块外面。如图18；
		另一种方式是，我们先直接把tokens以它们最初使用的嵌入向量输入进Self-Attention模块，然后，Self-Attention模块为输入序列的每个位置创建一个向量，再把该向量以某种方式或者加进Self-Attention模块的输入向量里，或者加进关注度权重向量里。在这种方式下，位置编码的产生位于Self-Attention模块内部，与Self-Attention模块的训练过程集成在一起。

##### 5.1 外置的位置编码与内嵌的位置编码

​		外置的位置编码：先把位置信息加入输入token的表达向量（比如通过预训练得到的词嵌入向量）里以后，再输入进Self-Attention模块。此时，位置向量的产生是在Self-Attention模块外面。

![img](http://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)

​		内嵌的位置编码：先直接把tokens以它们最初使用的嵌入向量输入进Self-Attention模块，然后，Self-Attention模块为输入序列的每个位置创建一个向量，再把该向量以某种方式或者加进Self-Attention模块的输入向量里，或者加进关注度权重向量里。在这种方式下，位置编码的产生位于Self-Attention模块内部，与Self-Attention模块的训练过程集成在一起。

##### 5.2 静态位置编码与动态位置编码

​		静态位置编码：使用确定性公式静态产生的位置编码向量，比如普通的Transformer用的就是这种类型的位置编码。如果使用外置的位置编码，那么位置编码向量可以直接按元素加进输入序列的词嵌入向量里，再输入进Self-Attention模块。

​		动态位置编码：这种位置编码是以可训练的变量的形式被创建的一个张量。当Self-Attention模块被训练时，这个位置向量也一同被训练。训练结束前，这个位置编码的值不是固定的。

​		无论是静态还是动态的位置编码，我们都可以把位置编码当作一个偏置向量，位置编码赋予输入序列中的不同位置的词以不同的偏置量。

​		不同位置的位置编码若完全独立，则与噪声无异——**位置编码应该有关联和规律**！
​		回想RNN，隐状态序列同时携带了输入序列的语法（位置）与语义（内容）信息，则**好的位置编码不应只与位置有关，还需要与该处词的内容有关****！**

==**还需要了解一下位置编码**==

#### 6 残差 Residuals

每个编码器中的每个子层（self-attention、FFN）都采用了残差网络中的short-cut结构，以解决深度学习中的退化问题。

<img src="http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" alt="img" style="zoom: 80%;" />

<div align = "center">图20 Encoder中的short-cut连接</div>

当然，残差网络也适用于Decoder。分别由两个编解码器组成的Transformer如图21所示：

![img](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

<div align = "center">图21 2 encoders and decoders</div>

##### 7 Encoder-Decoder Attention

在解码器中，Transformer block比编码器中多了个encoder-cecoder attention。在encoder-decoder attention中，$$Q$$来自于解码器的上一个输出，$$K$$和$$V$$则来自于编码器的输出。

##### 8 Linear 和Softmax层

Linear层为一层简单的全连接神经网络，输出维度与单词量一致（每个单元格对应一个单词的分数）。

假设模型从训练数据集学到10,000个英语单词（即输出词汇），则图中logits单元格个数为10,000个，不同单元格对应不同单词的分数。Softmax层将这些分数转化为概率，选择最高概率的单元格，并生成词典里对应位置的单词，则得到该时间步骤的输出。

<img src="http://jalammar.github.io/images/t/transformer_decoder_output_softmax.png" alt="img" style="zoom:80%;" />

<div align = "center">图22 Linear and Softmax</div>

##### 9 附论文《Attention is all you need》模型图片

<img src="https://pic1.zhimg.com/v2-0ffce3d357284dc546aea00971a0a77e_1440w.jpg?source=172ae18b" alt="论文笔记：Attention is all you need（Transformer）" style="zoom:50%;" />

注：Decoder中的Masked Multi-Head Attention 我是这么理解的：在解码器中，自注意层只允许注意输出序列中较早的位置。

---

## 专利创新点

在数据预处理（文档降噪、打标、数据增强、小样本、表格提取、摘要、文字提取、缺失数据、迁移学习、强化学习）方面，我们没有进行处理这几方面的铺垫工作，有创新性地进行数据预处理在短时间内有些困难。我们还是把重点放在对NER模型的改造上。

中英文的NER任务处理手段有所区别，中文涵盖的语义信息更为复杂，同时中文的字型、笔画、拼音等特征区别于英文的单词结构、前后缀。

在英文这个大前提条件下，Transformer有众多变种，但如果想适用于电网项目，想必Transformer的变种与针对中文的处理手段结合更好。

目前，虽然有针对中文的NER任务模型，但如果在此baseline模型基础上改造，想必无法想出更多的变种模型，所以==**改造思路为：英文NER任务的Transformer/Attention变种+NER任务针对中文独特的处理手段**==

### <font color='green'>英文Transformer\Attention变种模型</font>

#### 一、相对位置编码RPR(relative positional representation)

> 《self-attention with relative positional representation》来源：NAACL 2018

#### 二、Transformer-XL（extra long)

##### 创新点

- 片段递归机制（segment-level recurrence）

- 相对位置编码机制(relative positional encoding)

Transformer-XL带来的提升包括

-  捕获长期依赖的能力
- 解决了上下文碎片问题（context segmentation problem）
- 提升模型的预测速度和准确率。

##### 总结

Transformer由于自回归的特性，每个时间片的预测都需要从头开始，这样的推理速度限制了它在很多场景的应用。Transformer-XL提出的递归机制，使得推理过程以段为单位，段的长度越长，无疑提速越明显，从实验结果来看，Transformer-XL提速了300-1800倍，为Transformer-XL的使用提供了基础支撑。同时递归机制增加了Transformer-XL可建模的长期依赖的长度，这对提升模型的泛化能力也是很有帮助的。

仿照RPR，Transformer-XL提出了自己的相对位置编码算法，此编码 方法对比Transformer和RPR都有了性能的提升，而且从理论角度也有了可解释性。

### <font color='green'>针对中文的NER任务处理方法</font>

> 参考：http://www.4k8k.xyz/article/mumu_77zhl/109297207

近年来，基于词汇增强的中文NER主要分为2条主线：

- Dynamic Architecture：设计一个动态框架，能够兼容词汇输入；

- Adaptive Embedding ：基于词汇信息，构建自适应Embedding；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201026200138208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L211bXVfNzd6aGw=,size_16,color_FFFFFF,t_70#pic_center)

3.最终得出的结果：
1）引入词汇信息的方法，都相较于baseline模型biLSTM+CRF有较大提升，可见引入词汇信息可以有效提升中文NER性能。
2）采用相同词表对比时，FLAT和Simple-Lexicon好于其他方法。
3）结合BERT效果会更佳。

#### 一、Lattice LSTM

>  《Chinese NER Using Lattice LSTM》来源：2018 ACL

将词汇信息引入中文NER的开篇之作，作者将词节点编码为向量，并在字节点以注意力的方式融合词向量。

##### 总体架构

##### 创新点

#### 二、FLAT

> 《FLAT: Chinese NER Using Flat-Lattice Transformer》来源：2020 ACL

##### 背景

​		对于中文NER任务，词汇增强是有效提升效果的方法之一。LatticeLSTM是词汇增强的典型模型。但是这种Lattice结构，其模型结构比较复杂，并且由于lexicon word插入位置的动态性，导致LatticeLSTM模型无法并行，所以LatticeLSTM无法很好的利用GPU加速，其training以及inference的速度非常慢。所以怎么既能够很好的融入lexicon信息，同时又能够在不降低效果甚至提升效果的同时，大幅提升training以及inference的速度，这就是FLAT模型提出的背景。

​		目前在NER中，处理lattice结构的方式有两大类：1. 设计一个框架，能够兼容词汇信息的输入，典型的模型有：LatticeLSTM、LRCNN，这种模型的缺点是无法对长期依赖进行建模；2.将lattice结构转换为graph结构，典型的模型有：LGN、CGN，这种模型的缺点是：1.序列结构和graph结构还是有一定的差别；2.通常需要使用RNN来捕获顺序信息。而FLAT基于transformer，能够很好的解决上述的问题。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201026202734308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L211bXVfNzd6aGw=,size_16,color_FFFFFF,t_70#pic_center)

##### 创新点

本文在Lattice LSTM(ACL 2018)的基础上作出了两方面的改进：

- 提出了一种将Lattice图结构无损转换为**扁平的Flat结构**的方法，并将LSTM替换为了更先进的**Transformer Encoder**，该方法不仅弥补了Lattice LSTM无法「并行计算」(batchsize=1)的缺陷，而且更好地建模了序列的「长期依赖关系」；
- 提出了一种**针对Flat结构的「相对位置编码机制」**，使得字符与词汇得到了更充分更直接的信息交互，在基于词典的中文NER模型中取得了SOTA。

Self-Attention中，参考的是Transformer-XL的针对相对位置编码的self-attention。

#### 三、Simplify the Usage of Lexicon in Chinese NER

> 《Simplify the Usage of Lexicon in Chinese NER》来源：ACL 2020
>
> 论文：https://arxiv.org/pdf/1908.05969.pdf
>
> 参考：https://zhuanlan.zhihu.com/p/268558655

在词向量中嵌入词级信息

##### **ExSoftword Feature**

![img](https://ivenwang.com/wp-content/uploads/2020/11/1-2-1024x504.png)

拿图中的中山西路举例，上下两行圆角框框是两种匹配（类似分词）的结果。以下面的一个结果为例，山这个字，在“中山”，“中山西”，“山西路”这三个词中分别作为开始（B）、中间（M）、结尾（E）出现，那么它的特征向量就是 {1,1,1,0,0}，这里五个位置就代表 BMESO 的 POS tagging。同理，“西”的就是 {0,0,1,0,0}。

这种特征表示的方法并不好：比如“西”作为 M 出现在“中山西路”和“山西路”两次，这就没有在向量中体现出来。

**SoftLexicon**

![img](https://ivenwang.com/wp-content/uploads/2020/11/2-1-1024x503.png)

>这种方法对于每个字都定义了四个集合，分别代表这个字作为 B/M/E/S 在哪些词中出现。如图所示，“山”在“山西”中作为开头，因此“山”的 B 集合就是 {“山西”}。如果这个字没有作为某种 POS 在词中出现，就定义这个字的这个 POS 的集合为 {“None”}。

通过给集合里每个词的词向量加权求和，得到集合的向量，拼接四个向量，就得到了这个字的词级别特征表示。这里加权求和的“权”是词语在语料中的词频。

得到字的词级别特征向量之后，直接拼接到字向量之后，后面可以输入任意 NER 的模型使用。

1.首先对于一个输入句子s的一个字符，对它的所有匹配词做分为BMES四个类，得到4个词集合；

2.接着对词集合做压缩，主要是把每个类别的word embedding压缩为一个embedding，这里有两种方法：词加权和词平均。

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7Bv%7D%5E%7Bs%7D%28%5Cmathcal%7BS%7D%29%3D%5Cfrac%7B1%7D%7B%7C%5Cmathcal%7BS%7D%7C%7D+%5Csum_%7Bw+%5Cin+%5Cmathcal%7BS%7D%7D+%5Cboldsymbol%7Be%7D%5E%7Bw%7D%28w%29)和![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7Bv%7D%5E%7Bs%7D%28S%29%3D%5Cfrac%7B4%7D%7BZ%7D+%5Csum_%7Bw+%5Cin+S%7D+z%28w%29+%5Cboldsymbol%7Be%7D%5E%7Bw%7D%28w%29)，两者都使用每个词在一个静态数据集上出现的频率作为权重，是一个静态权重；

3.最后将词典信息合并到字符表示上

##### 相关结论图表

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20211111163827921.png" alt="image-20211111163827921" style="zoom:67%;" />

<img src="C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20211111163850242.png" alt="image-20211111163850242" style="zoom:80%;" />

![image-20211111163905451](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20211111163905451.png)

![image-20211111163915752](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20211111163915752.png)

## 总体模型

#### <font color='green'>Embedding层</font>

> NER是一个重底层的任务，我们应该集中精力在embedding层下功夫，引入丰富的特征。如char、bigram、词典特征、词性特征、偏旁部首甚至拼音等。底层特征越丰富、差异化越大，效果越好。

多尺度特征融合！！

