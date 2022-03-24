## 预训练模型（Pre-trained Models)

> 《Pre-trained Models for Natural Language Processing: A Survey》邱锡鹏
>
> 参考网站：
>
> http://xtf615.com/2020/11/28/Pretrain-Models/
>
> https://github.com/loujie0822/Pre-trained-Models
>
> https://zhuanlan.zhihu.com/p/49271699

**CV领域中的预训练：**

>  用训练集合数据对网络进行预先训练，学习模型参数，有利于网络结构相似模型的参数初始化（其他参数则随机初始化），减小训练时间。

**CV中为什么需要预训练？**

1.当网络结构很深，参数众多的情况下，如果训练数据少，很难训练如此复杂的网络。

2.即使训练数据充足的情况下，预训练能加快收敛速度。

3.细化来说，对于层级的CNN结构来说，不同层级的神经元学习到不同的图像特征。比如，人脸识别任务中，最底层神经元学习到线段等特征，第二个隐层学习到人脸五官的轮廓，第三层学到的是人脸的轮廓。越往上抽取的特征与任务越相关。因此，使用预训练方法初始化部分模型参数具备通用性。

**反观NLP，Word Embedding就是早期的预训练技术应用于NLP任务的产物。**

---

### NNLM（Neural Network Language Model)

<img src="https://pic2.zhimg.com/v2-e2842dd9bc442893bd53dd9fa32d6c9d_r.jpg" alt="preview" style="zoom:150%;" />

如图，NNLM最底层输入为One-Hot表示的单词向量，每个单词与矩阵$$C$$做矩阵乘法，得到每个单词的Embedding表达$$C(W_t)$$。矩阵$$C$$为右侧给出的矩阵，行数为词典中字的个数，每一行表示该字对应的向量。这些向量由随机初始化产生，通过模型训练不断学习，最终形成字典中每个字的Embedding向量。位置$$1:t-1$$的$$C(W_t)$$进行拼接，通过tanh激活函数后，接一个残差结构，再通过Softmax激活函数，得到当前$$i$$位置的单词，达到预测的效果。

> **什么是语言模型？**
>
> **任务**：用输入数据的一部分信息以某种形式去预测另一部分信息
>
> **数学描述：**
>
> 给定文本序列，$$x_{1:T} = [x_1, x_2,...,x_T]$$，其联合概率$$p(x_{1:T})$$可被分解为：
> $$
> p(x_{1:T}) = \prod^T_{t=1}p(x_t|x_{0:t-1}), x_t \in V
> $$
> 其中，$$x_0$$是特殊的token，用于标识句子的开头，V是词典
>
> 链式法则中的每个部分$$p(x_t|x_{0:t-1})$$是给定上下文$$x_{0:t-1}$$条件下，当前要预测的词$$x_t$$在整个词典上的条件概率分布。这意味着当前的单词只依赖于前面的单词，即单向的或者自回归的。
>
> **具体实现方面：**
>
> 对于上下文$$x_{0:t-1}$$，可以采用神经编码器$$f_{enc}()$$来进行编码，然后通过一个预测层来预测单词$$x_t$$的条件概率分布，形式化的：
> $$
> p(x_t|x_{0:t-1}) = g_{LM}(f_{enc}(x_{0:t-1}))
> $$
> 其中，$$g_{LM}$$是预测层（如softmax全连接层）
>
> **延伸**：LM的缺点在于，除了本身的信息之外，**每个单词的编码只融入了其所在句子左侧的上下文单词的信息**。而实际上，每个单词左右两侧上下文信息都是非常重要的。为什么不能同时融入左右两侧的单词信息呢？主要是因为我们的学习目标是预测下一个词，如果让当前词同时融入两侧的信息，会造成标签的leak问题。解决这个问题的方法是采用bidirectional LM (Bi-LM)即：分别考虑从左到右的LM和从右到左的LM，这两个方向的LM是分开建模的。也就是说，训练过程中，不会在一个LM预测下一个词的时候，用到另一个LM的encode的信息。最后训练完成后，每个单词可以把两个$$f_{left-enc}$$和$$f_{right-enc}$$的输出拼接在一起来形成最终的表征。

**语言模型的目标是预测下一位置的单词，但同时有一个副产品，就是模型学习到的矩阵$$C$$，该矩阵里存的就是单词的向量表达Embedding。**

学习到单词对应的Embedding后，后续任务如何使用呢？Embedding能被看做预训练吗？![preview](https://pic3.zhimg.com/v2-5875b516b8b3d4bad083fc2280d095fa_r.jpg)

>  **NLP任务可大致分为四种类型：**
>
> 1.序列标注：分词、POS、NER等；
>
> 2.句子关系判断：Entailment、Q&A、自然语言推理；
>
> 3.分类任务：文本分类，情感分类；
>
> 4.生成式任务：机器翻译、文本摘要。

图中的任务为问答任务，给出一个Question，判断一句话是否为该问题的正确答案。最底层，文本序列的One-Hot编码作为原始输入，与预训练好的Embedding矩阵$$C$$作矩阵乘法将其映射为embedding向量，随后经过隐藏层训练，后续比较Question和各Answer的匹配程度，判断哪句话是正确答案。

模型中，Word Embedding Matrix矩阵是预训练的结果，否则还需要初始化矩阵参数，随着模型迭代训练。这与CV中的预训练有共通之处：初始化参数，加快模型收敛速度。

#### **NNLM的局限**

多义词问题。Word Embedding对单词进行编码时，往往区分不开多义词。问题在于，同一个单词占同一行参数空间，多义对应的不同上下文信息都会编码到相同的embedding空间中去。所以word embedding无法区分多义词的不同语义。

---

### ELMo(Embedding from Language Models)

提出ELMo的论文题目为：Deep contextualized word representation。在此之前的Word Embedding本质上是**静态**的方式，embedding向量**不随上下文变化而变化**，不管这个词位于哪个句子中，词的表示都是唯一的。此类预训练模型致力于学习word embedding本身，**不考虑上下文信息，只关注词本身的语义**。代表性工作包括：**NNLM**[[2\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-2)，**word2vec**[[3\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-3)（CBOW， Skip-Gram），**GloVe**[[4\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-4)等。

所谓静态，是指训练好之后每个单词的表达固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的Word Embedding不会随着上下文场景变化而变化。面对具体场景时，即使从上下文中可以判断单词$$w$$为某一语义s，但是s对应的word embedding内容也不会因此而改变。

**ELMo的本质思想是**：事先用语言模型学好一个单词的word embedding，此时向量包含了不同语义的特定上下文，无法区分多义词。在实际使用word embedding时，结合上下文单词的语义生成word embedding表示，这样经过调整后的word embedding更能表达在这个上下文中的具体含义，自然也就解决多义词的问题。

#### ELMo预训练阶段

<img src="https://pic4.zhimg.com/80/v2-fe335ea9fdcd6e0e5ec4a9ac0e2290db_720w.jpg" alt="img" style="zoom:150%;" />

ELMO采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的Word Embedding作为新特征补充到下游任务中。

上图展示的是其预训练过程，它的网络结构采用了双层双向LSTM，目前语言模型训练的任务目标是根据单词$$W_i$$的上下文正确预测当前位置单词$$W_i$$，之前的单词序列称为上文，之后的单词序列称为下文。图中左端的前向双层LSTM代表正方向编码器，输入的是从左到右顺序的$$W_{0:i-1}$$；右端的逆向双层LSTM代表反方向编码器，输入的是从右往左的逆序的$$W_{i+1:N}$$。

训练完成后，输入一个新句子，句中每个单词都能得到对应的三个Embedding:最底层是单词的Word Embedding，往上走是第一层双向LSTM中对应单词位置的Embedding，这层编码单词的**句法信息**更多一些；再往上走是第二层LSTM中对应单词位置的Embedding，这层编码单词的**语义信息**更多一些。也就是说，ELMO的预训练过程不仅仅学会单词的Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。

#### ELMo Fine-Tuning阶段

<img src="https://pic2.zhimg.com/80/v2-ef6513ff29e3234011221e4be2e97615_720w.jpg" alt="img" style="zoom:200%;" />

如我们的下游任务仍然是QA问题，此时对于问句X，我们可以先将句子X作为预训练好的ELMO网络的输入，这样句子X中每个单词在ELMO网络中都能获得对应的三个Embedding，之后给予这三个Embedding中的每一个Embedding一个权重a，这个权重可以学习得来，根据各自权重累加求和，将三个Embedding整合成一个。然后将整合后的这个Embedding作为X句在自己任务的那个网络结构中对应单词的输入，以此**作为补充的新特征**给下游任务使用。

#### ELMo局限

1.特征抽取方面，ELMo使用了LSTM而不是基于注意力机制的Transformer。

> 根据架构不同，可以分为三种。
>
> - **卷积模型**：通过卷积操作来汇聚目标词的邻居的局部信息，从而捕获目标词的语义。**优点**在易于并行计算，且能够很捕获局部上下文信息，**但**面对长距离序列关系时存在天然缺陷。典型工作是EMNLP 2014的文章**TextCNN**[[13\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-13)
>
> - **序列模型** (Sequential models)：以序列的方式来捕获词的上下文信息。如LSTMs、GRUs。实践中，通常采取bi-directional LSTMs或bi-directional GRUs来同时捕获目标词双向的信息。**优点**在于能够捕获整个语句序列上的依赖关系，**缺点**是捕获的长距离依赖较弱，最主要的是并行计算能力弱。典型工作是NAACL 2018的文章：**ELMo**[[6\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-6)。
>
> - **图模型** (Graph-based models)：将词作为图中的结点，通过预定义的词语之间的语言学结构（e.g., 句法结构、语义关系等）来学习词语的上下文表示。缺点是，构造好的图结构很困难，且非常依赖于专家知识或外部的nlp工具，如依存关系分析工具。典型的工作如：NAACL 2018上的工作[[15\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-15)。
>
>   Transformer实际上是图模型的一种特例。句子中的词构成一张全连接图，图中任意两个词之间都有连边，连边的权重衡量了词之间的关联，通过self-attention来动态计算，目标是让模型自动学习到图的结构（实际上，图上的结点还带了词本身的属性信息，如位置信息等）。

2.ELMo左端编码器输入为当前单词wordembedding及上文而没有下文，右端编码器输入为当前单词的wordembedding及下文而没有上文，两个方向的LM是独立训练的，只有word embedding和Softmax共享参数，并没有达到所述的"Deeply bidirecitonal"。所以，每个LM在预测下一单词时无法利用另一方向的信息，这一点和BERT使用的MLM（带掩码的语言模型，Masked Language Model）不同。

---

### GPT（Generative Pre-Training)

GPT采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过Fine-tuning的模式解决下游任务。与ELMo不同的是：1.特征抽取器用的是Transformer而不是RNN；2.GPT的预训练虽然仍然是以语言模型作为目标任务，但是采用的是单向的语言模型（只利用了上文，没有利用下文，丢掉了很多信息）。

<img src="https://pic1.zhimg.com/80/v2-5028b1de8fb50e6630cc9839f0b16568_720w.jpg" alt="img" style="zoom: 200%;" />

<img src="https://pic3.zhimg.com/80/v2-587528a22eff055b6f479dae67f7c1aa_720w.jpg" alt="img" style="zoom:200%;" />

GPT属于{上下文感知的词嵌入}中的{微调}模型，而上文ELMo属于{上下文感知的词嵌入}中的{仅作为特征提取器}，NNLM则属于{非上下文感知的词嵌入}。

GPT属于{上下文感知的词嵌入}中的{微调}模型，也就是说，使用GPT进行预训练的模型结构需要向GPT看齐，在训练时改变GPT结构的具体参数，以使该网络更适合手头的任务。

对于NLP不同类的任务，GPT预训练模型如何才能改造下游任务呢？

<img src="https://pic1.zhimg.com/80/v2-4c1dbed34a8f8469dc0fefe44b860edc_720w.jpg" alt="img" style="zoom:200%;" />

---

## 预训练模型分类

![category](http://xtf615.com/picture/machine-learning/ptm_category.png)

### 根据表征类型分类

根据表征类型的不同可以分为：非上下文感知的表征(Non-Contextual Representation)和上下文感知的表征(Contextual Representation)。

- **非上下文感知的词嵌入**。（浅层词嵌入）

  静态的，不随上下文变化而变化，不管这个词位于哪个句子中，词的表示都是唯一的。此类预训练模型致力于学习word embedding本身，不考虑上下文信息，只关注词本身的语义。代表性工作包括：**NNLM**[[2\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-2)，**word2vec**[[3\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-3)（CBOW， Skip-Gram），**GloVe**[[4\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-4)等。

  缺点：1.无法解决一词多义等复杂问题。2.这一类词嵌入方法通常采用浅层网络进行训练，应用于下游任务时，整个模型的其他部分仍需要从头开始学习（与预训练这一说法相悖）。

- **上下文感知的词嵌入**。（深层词嵌入）

  动态的，词嵌入会随着词所在的上下文不同而动态变化，能够解决一词多义问题。此类预训练模型致力于学习依赖上下文的word embeddings。他们将此前的word-level拓展到sentence-level或者更高层次。

  给定文本$$x_1,x_2,...,x_T$$，为了形成上下文感知的嵌入，$$x_t$$的表示需要依赖于整个文本。 即: $$[h_1,...,h_T]=f_{enc}(x_1,...,x_T)$$。$$h_t$$称为token $$x_t$$的上下文感知的词嵌入或动态词嵌入，因为其融入了整个文本中的上下文信息。$$f_{enc}$$为Encoder编码器，把一个向量空间映射为另一个向量空间。

  <img src="http://xtf615.com/picture/machine-learning/word_embeddings.png" alt="word_embeddings" style="zoom:50%;" />

  根据{词嵌入是否在下游任务训练过程中变化}，上下文感知的词嵌入可以分为两类：

  - **仅作为特征提取器**（feature extractor）

    特征提取器产生的上下文词嵌入表示，在下游任务训练过程中是**固定不变**的。相当于只是把得到的上下文词嵌入表示喂给下游任务的模型，作为补充的特征，只学习下游任务特定的模型参数。

    代表性工作包括：

    (1) **CoVe**[[5\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-5)。用带注意力机制的seq2seq从机器翻译任务中预训练一个LSTM encoder。输出的上下文向量(CoVe)有助于提升一系列NLP下游任务的性能。

    (2) **ELMo**[[6\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-6)。 用两层的Bi-LSTM从双向语言模型任务BiLM（包括1个前向的语言模型以及1个后向的语言模型）中预训练一个Bi-LSTM Encoder。能够显著提升一系列NLP下游任务的性能。

  - **微调** (fine-tune)

    在下游任务中，上下文编码器的参数也会进行微调。即：把预训练模型中的encoder模型结构都提供给下游任务，这样下游任务可以对Encoder的参数进行fine-tune。

    代表性工作包括：

    (1) **ULMFiT**[[7\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-7) (Universal Language Model Fine-tuning)： 通过在文本分类任务上微调预训练好的语言模型达到了state-of-the-art结果。这篇也被认为是预训练模型微调模式的开创性工作。提出了3个阶段的微调：在通用数据上进行语言模型的预训练来学习通用语言特征；在目标任务所处的领域特定的数据上进行语言模型的微调来学习领域特征；在目标任务上进行微调。文中还介绍了一些微调的技巧，如区分性学习率、斜三角学习率、逐步unfreezing等。

    (2) **GPT[[8\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-8)**(Generative Pre-training) ：使用单向的Transformer预训练单向语言模型。单向的Transformer里头用到了masked self-attention的技巧（相当于是Transformer原始论文里头的Decoder结构），即当前词只能attend到前面出现的词上面。之所以只能用单向transformer，主要受制于单向的预训练语言模型任务，否则会造成信息leak。

    (3) **BERT** [[9\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-9)(Bidirectional Encoder Representation from Transformer)：使用双向Transformer作为Encoder（即Transformer中的Encoder结构)，引入了新的预训练任务，带mask的语言模型任务MLM和下一个句子预测任务NSP。由于MLM预训练任务的存在，使得Transformer能够进行双向self-attention。

### 根据预训练任务分类

**NLP任务根据数据源可分为3种学习方式：**

- **监督学习**：从”输入-输出pair”监督数据中，学习输入到输出的映射函数。
- **无监督学习**：从无标签数据中学习内在的知识，如聚类、隐表征等。
- **自监督学习**：监督学习和无监督学习的折中。训练方式是监督学习的方式，但是输入数据的标签是模型自己产生的。核心思想是，用输入数据的一部分信息以某种形式去预测其另一部分信息。例如BERT中使用的MLM就是属于这种，输入数据是句子，通过句子中其它部分的单词信息来预测一部分masked的单词信息。

在nlp领域，除了机器翻译存在大量的监督数据，能够采用监督学习的方式进行预训练以外（例如CoVe利用机器翻译预训练Encoder，并应用于下游任务），大部分预训练任务都是使用自监督学习的方式。

---

**预训练可细分为五种任务**

语言模型（LM）、带掩码的语言模型（MLM）、排列语言模型（PLM）、降噪自编码器（DAE）、对比学习（CTL）

#### 语言模型（LM）（补充）



#### 带掩码的语言模型（MLM）

MLM主要是从BERT开始流行起来的，能够解决单向的LM的问题，进行双向的信息编码。

MLM就好比英文中的完形填空问题，需要借助语句/语篇所有的上下文信息才能预测目标单词。具体的做法就是随机mask掉一些token，使用特殊符号[MASK]来替换真实的token，这个操作相当于告知模型哪个位置被mask了，然后训练模型通过其它没有被mask的上下文单词的信息来预测这些mask掉的真实token。

具体实现时，实际上是个多分类问题，将masked的句子送入上下文编码器Transformer中进行编码，[MASK]特殊token位置对应的最终隐向量输入到softmax分类器进行真实的masked token的预测。

损失函数：
$$
L_{MLM} = -\sum_{\hat{x} \in m(x)}logp(\hat{x}|x_{\backslash m(x)})
$$
其中，$$m(x)$$表示句子$$x$$中被mask掉的单词集合；$$x_{\backslash m(x)}$$是除了masked单词之外的其它单词。

##### **MLM的缺点有几大点：**

- 会造成pre-training和fine-tuning之间的gap。在fine-tuning时是不会出现pre-training时的特殊字符[MASK]。

> 为了解决这个问题，作者对mask过程做了调整，即：在随机挑选到的15%要mask掉的token里头做了进一步处理。其中，**80%使用[MASK] token替换目标单词；10%使用随机的词替换目标单词；10%保持目标单词不变**。除了解决gap之外，还有1个好处，即：预测一个词汇时，模型并不知道输入对应位置的词汇是否为正确的词 (10%概率)，这就迫使模型更多地依赖于上下文信息去预测目标词，并且赋予了模型一定的纠错能力。

- MLM收敛的速度比较慢，因为训练过程中，一个句子只有15%的masked单词进行预测。
- MLM不是标准的语言模型，其有着自己的独立性假设，即假设mask词之间是相互独立的。
- 自回归LM模型能够通过联合概率的链式法则来计算句子的联合概率，而MLM只能进行联合概率的有偏估计(mask之间没有相互独立)。

##### **MLM变体**

- **Sequence-to-Sequence MLM(Seq2Seq MLM)**

将MLM分类任务变成seq2seq序列自回归预测任务，采用encoder-decoder的方式。原始的语句中有一段**连续出现的单词**被mask掉了。encoder的输入是masked的句子，decoder以自回归的方式来依次地预测masked tokens。

这种预训练任务很适合用于**生成式任务**。代表性工作有：微软的 **MASS**[[16\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-16) 和 Google的**T5**[[17\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-17) 。这种预训练认为能够有效提高seq2seq类型的下游任务的表现。其损失函数为：
$$
L_{S2SMLM}=-\sum^j_{t=i}logp(x_t|x_{\backslash x_{i:j}},x_{i:t-1})
$$

$$x_{i:j}$$是句子$$x$$被masked的n-gram span，是连续出现的单词。基于encoder端的输入序列￥$$x_{\backslash x_{i:j}}$$以及decoder已经解码的部分$$x_{i:t-1}$$来自回归地预测下一个时间步$$t$$的单词。

- **Enhanced MLM (E-MLM)**：增强版MLM。
  - **RoBERTa**[[18\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-18)：Facebook 2019提出的方法。改进了BERT种静态masking的方法，采用了动态masking的方法。
  - **UniLM**[[19\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-19)： 微软提出的方法。UniLM拓展mask prediction任务到三种语言模型任务中，单向预测、双向预测、seq2seq预测。
  - **XLM**[[20\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-20): 将MLM应用到翻译语言模型中，即“双语料句子对“构成一个句子，然后使用MLM。
  - **SpanBERT**[[21\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-21)：Facebook提出的方法。改进了BERT中掩码最小单元为token所导致的强相关字词被割裂开来的问题，使用了span masking来随机掩盖一段连续的词。同时额外提出了一种边界学习目标 (Span Boundary Objective) ，希望被掩盖的词能够融入边界的信息，即基于边界之外的两个单词的向量和masked单词的位置向量来预测masked单词。这个改进对于抽取式问答任务有很大的帮助。
  - **ERNIE**[[22\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-22)：百度提出的ERNIE，将外部知识融入到MLM中。引入了命名实体Named Entity外部知识来掩盖实体单元，进行训练。

#### 排列语言模型（PLM）

PLM在XLNet[[23\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-23)中被提出。动机来源主要在于上文阐述的MLM的几大缺点，如预训练和微调阶段的gap，mask词之间的独立性假设等。

在传统的单向自回归语言模型LM中，句子的联合概率因子分解是按照从左到右或者从右到左的方式分解成条件概率的链式乘积的，这可以看作是其中两种联合概率的因子分解序。实际上，句子的联合概率的因子分解序还可以有很多种，可以任意的排列组合进行因子分解。PLM就是对联合概率进行因子分解得到排列，分解得到的排列只决定了模型自回归时的预测顺序，不会改变原始文本序列的自然位置。即：PLM只是针对语言模型建模不同排列下的因子分解排列，并不是词的位置信息的重新排列。

那么为什么这种方式每个位置能够编码原始语句中双向的上下文信息呢? 

首先，前提是，模型的参数在所有的分解序下是共享的。其次，在每种因子分解序对应的排列语句下，对某个位置，会编码排列句子中出现在该位置前面的其它词的信息；那么在所有的因子分解下，每个词都有机会出现在该位置的左侧，那么总体上该词就会编码所有词的信息。

理想优化的目标是所有因子分解序得到的排列上的期望对数似然。

![image-20220302214453941](C:\Users\27645\AppData\Roaming\Typora\typora-user-images\image-20220302214453941.png)

进一步，强调下实现上的亮点。实际实现的过程中，仍然采用原始输入语句，即保持原始句子的自然序，而模型内部会自动进行排列操作，对transformer进行适当的attention mask操作就能达到在因子分解序上进行自回归预测的目的。然而，预测的时候如果没有考虑目标词在原始序列中的位置信息的话，会导致预测的目标词不管在排列句子中的哪个位置，其分布都是一样的（虽然输入语句自然序不变，但是建模的时候不进行考虑的话，相当于对随机扰动的序列进行建模预测，直观上感觉这样显然无效）。作者做了改进，在预测$$x_{z_t}$$词本身时，要利用到其在原始句子的位置编码信息，即：$$p_\theta(x_{z_t}|x_{z<t},z_t)$$，即target-position-aware的next-token预测 (这样就能在排列句上预测过程中，时刻兼顾原始输入语句序)。但是为了实现自回归预测，transformer在编码的时候不能把目标预测词本身的内容编码进目标词对应位置的隐状态中，而只能使用目标预测词的位置信息；而目标预测词之前的其它词就要考虑其本身的内容信息。每个位置的词都有两种隐状态向量，因此需要做这两种区分，是使用ztzt位置信息还是其对应的内容信息$$x_{z_t}$$。为了方便实现该操作，作者采用了two-stream self-attention。

#### 降噪自编码器（DAE）

DAE在原始文本上加了噪声，即corrupted input作为输入，目标是基于corrupted input来恢复原始的文本。MLM属于DAE的一种形式，除此之外DAE还有其它的形式。下面的这些细类别，综述参考的是Facebook2019的文章BART[[24\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-24)。

- **Token masking**：随机抽样token，并用[MASK] 特殊字符替换。BERT属于这种。
- **Token Deletion**：随机删除token。和masking的区别在于，模型还需要预测被删除的单词的真实位置。
- **Text Infilling**：连续的一些token被替换为单一的[MASK]，模型需要进行缺失文本的填充。和SpanBERT比较像，区别在于SpanBert mask 掉几个词就填上几个mask ，在这里作者mask掉的span 都只是填上一个mask, 目的是为了让模型自己去学习多少个token被mask了。
- **Sentence Permutation**：对文档的语句顺序进行随机扰动。
- **Document Rotation:** 随机选择某个token，然后让文档进行rotation从而使得被选中的词作为第一个词（例如：12345,选中3时，变成34512)，这样是为了让模型能够识别文档的真正起始词。

#### 对比学习（CTL）

前面介绍的方法主要是基于上下文的PTMs，即：基于数据本身的上下文信息构造辅助任务。这里作者介绍的另一大类的预训练方法是基于对比的方法，即：通过对比来进行学习。很像learning to rank中的pairwise方法。CTL全称：Contrastive Learning，假设了观测文本对之间的语义比随机采样的文本对之间的语义更近。

基于对比的方法主要包括如下一些具体的预训练任务类型，只不过下面这些对比的方法和上面的优化目标在形式上差异挺大的。

- **Deep InfoMax**：最大化整体表示和局部表示之间的互信息。代表性工作是ICLR2020的 **InfoWord**[[25\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-25): 最大化一个句子的全局表征和其中一个ngram的局部表征之间的Mutual Information。
- **Replaced Token Detection (RTD):** 给定上下文条件下，预测某个token是否被替换。这里头，可能“对比体现在要让模型去学习替换前和替换后的区别。在RTD任务中，和MLM不同的是，输入序列中所有的token都能够得到训练和预测，因此比较高效，同时能解决[MASK] token带来的预训练和fine-tuning之间的gap。代表性方法google在ICLR2020提出的**ELECTRA**[[26\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-26)。ELECTRA利用基于MLM的generator来对句子的某些token进行合理的替换，然后用discriminator来判断这些token是真实的，还是被generator替换了。最后预训练完，只保留discriminator进行下游任务的fine-tuning。另一个代表性工作是facebook在ICLR2020提出的**WKLM**[[27\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-27)，替换的时候是entity-level而不是token-level。具体而言，将entity替换为同类型的其它entity，然后训练模型进行判断。
- **Next Sentence Prediction (NSP)：** 判断文档中的两个句子是不是连在一起的。即连贯性判断任务。采样的时候，对某个句子，50%的时候其真实的句子作为下一句；50%的时候随机选择某个句子作为下一句，模型需要理解句子之间的关系。出发点是希望能够理解句子对之间的关系，这也是BERT中的第二个预训练任务。可惜的是，这个任务被其它工作怼的很惨，基本上的结论都是用处不大。可能的原因在ALBERT中有解释，大概就是随机选择句子太trivial了，只需要基于句子对之间的主题是否一致就能够把负样本判断出来，而不是句子间的连贯性判断。相当于，只需要判断句子对是来自同一篇文档里头(相同主题)的还是来自不同篇文档里头的，而我们的目的是对同一篇文档里头，句子之间的连贯性也能够很好的判断。显然，NSP任务需要做一定的改进。
- **Sectence Order Prediction (SOP):** **ALBERT**[[28\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-28) 中提出的预训练任务。具体而言，将同一篇文档里头的连贯句子对作为正样本，而把连贯的句子对的顺序颠倒后作为负样本。这样就能强迫模型真正的学习到句子之间的连贯性，而不仅仅是通过句子之间的主题差异来判断的。另外，阿里的**StructBERT**[[29\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-29) 也采用了该任务。

<img src="http://xtf615.com/picture/machine-learning/summary_table.png" alt="summar" style="zoom:200%;" />



## 预训练延伸方向

- 基于**知识增强**的预训练模型，Knowledge-enriched PTMs
- **跨语言或语言特定的**预训练模型，multilingual or language-specific PTMs
- **多模态**预训练模型，multi-modal PTMs
- **领域特定**的预训练模型，domain-specific PTMs
- **压缩**预训练模型，compressed PTMs

#### 基于知识增强的预训练模型

PTMs主要学习通用语言表征，但是缺乏领域特定的知识。因此可以考虑把外部的知识融入到预训练过程中，让模型同时捕获上下文信息和外部的知识。早期的工作主要是将知识图谱嵌入和词嵌入一起训练。从BERT开始，涌现了一些融入外部知识的预训练任务。代表性工作如：

- **SentiLR**[[31\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-31) : 引入word-level的语言学知识，包括word的词性标签(part-of-speech tag)，以及借助于SentiWordNet获取到的word的情感极性(sentiment polarity)，然后将MLM拓展为label-aware MLM进行预训练。包括：给定sentence-level的label，进行word-level的知识的预测 (包括词性和情感极性); 基于语言学增强的上下文进行sentence-level的情感倾向预测。作者的做法挺简单的，就是把sentence-level label或word-level label进行embedding然后加到token embedding/position embedding上，类似BERT的做法。然后，实验表明该方法在下游的情感分析任务中能够达到state-of-the-art水平。

- **ERNIE (THU)[[32\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-32) :** 将知识图谱上预训练得到的entity embedding融入到文本中相对应的entity mention上来提升文本的表达能力。具体而言，先利用TransE在KG上训练学习实体的嵌入，作为外部的知识。然后用Transformer在文本上提取文本的嵌入，将文本的嵌入以及文本上的实体对应的KG实体嵌入进行异构信息的融合。学习的目标包括MLM中mask掉的token的预测；以及mask文本中的实体，并预测KG上与之对齐的实体。

  类似的工作还包括KnowBERT, KEPLER等，都是通过实体嵌入的方式将知识图谱上的结构化信息引入到预训练的过程中。

- **K-BERT**[[33\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-33) : 将知识图谱中与句子中的实体相关的三元组信息作为领域知识注入到句子中，形成树形拓展形式的句子。然后可以加载BERT的预训练参数，不需要重新进行预训练。也就是说，作者关注的不是预训练，而是直接将外部的知识图谱信息融入到句子中，并借助BERT已经预训练好的参数，进行下游任务的fine-tune。这里头的难点在于，异构信息的融合和知识的噪音处理，需要设计合适的网络结构融合不同向量空间下的embedding；以及充分利用融入的三元组信息（如作者提到的soft position和visible matrix）。

####  跨语言或语言特定的预训练模型

这个方向主要包括了跨语言理解和跨语言生成这两个方向。

对于跨语言理解，传统的方法主要是学习到多种语言通用的表征，使得同一个表征能够融入多种语言的相同语义，但是通常需要对齐的弱监督信息。但是目前很多跨语言的工作不需要对齐的监督信息，所有语言的语料可以一起训练，每条样本只对应一种语言。代表性工作包括：

- **mBERT**[[34\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-34) ：在104种维基百科语料上使用MLM预训练，即使没有对齐最终表现也非常不错，没有用对齐的监督信息。
- **XLM**[[20\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-20)：在mBERT基础上引入了一个翻译任务，即：目标语言和翻译语言构成的双语言样本对输入到翻译任务中进行对齐目标训练。这个模型中用了对齐的监督信息。
- **XLM-RoBERTa**[[35\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-35)：和mBERT比较像，没有用对齐的监督信息。用了更大规模的数据，且只使用MLM预训练任务，在XNLI, MLQA, and NER.等多种跨语言benchmark中取得了SOA效果。

对于跨语言生成，一种语言形式的句子做输入，输出另一种语言形式的句子。比如做机器翻译或者跨语言摘要。和PTM不太一样的是，PTM只需要关注encoder，最后也只需要拿encoder在下游任务中fine-tune，在跨语言生成中，encoder和decoder都需要关注，二者通常联合训练。代表性的工作包括：

- **MASS**[[16\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-16)：微软的工作，多种语言语料，每条训练样本只对应一种语言。在这些样本上使用Seq2seq MLM做预训练。在无监督方式的机器翻译上，效果不错。
- **XNLG**[[36\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-36)：使用了两阶段的预训练。第一个阶段预训练encoder，同时使用单语言MLM和跨语言MLM预训练任务。第二个阶段，固定encoder参数，预训练decoder，使用单语言DAE和跨语言的DAE预训练任务。这个方法在跨语言问题生成和摘要抽取上表现很好。

#### 多模态预训练模型

多模态预训练模型，即：不仅仅使用文本模态，还可以使用视觉模态等一起预训练。目前主流的多模态预训练模型基本是都是文本+视觉模态。采用的预训练任务是visual-based MLM，包括masked visual-feature modeling and visual-linguistic matching两种方式，即：视觉特征掩码和视觉-语言语义对齐和匹配。这里头关注几个关于image-text的多模态预训练模型。这类预训练模型主要用于下游视觉问答VQA和视觉常识推理VCR等。

- **双流模型**：在双流模型中文本信息和视觉信息一开始先经过两个独立的Encoder（Transformer）模块，然后再通过跨encoder来实现不同模态信息的融合，代表性工作如：NIPS 2019, **ViLBERT**[[37\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-37)和EMNLP 2019, **LXMERT**[[38\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-38)。
- **单流模型**：在单流模型中，文本信息和视觉信息一开始便进行了融合，直接一起输入到Encoder（Transformer）中，代表性工作如：**VisualBERT** [[39\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-39)，**ImageBERT**[[40\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-40)和**VL-BERT** [[41\]](http://xtf615.com/2020/11/28/Pretrain-Models/#refer-41)。

#### 模型压缩方法

预训练模型的参数量过大，模型难以部署到线上服务。而模型压缩能够显著减少模型的参数量并提高计算效率。压缩的方法包括：

- **剪枝（pruning）：**去除不那么重要的参数（e.g. 权重、层数、通道数、attention heads）
- **量化（weight quantization）：**使用占位更少（低精度）的参数
- **参数共享（parameter sharing）：**相似模型单元间共享参数
- **知识蒸馏（knowledge diistillation）：**用一些优化目标从原始的大型teacher模型中蒸馏出一个小的student模型。通常，teacher模型的输出概率称为soft label，大部分蒸馏模型让student去拟合teacher的soft label来达到蒸馏的目的。蒸馏之所以work，核心思想是因为好模型的目标不是拟合训练数据，而是学习如何泛化到新的数据。所以蒸馏的目标是让学生模型学习到教师模型的泛化能力，理论上得到的结果会比单纯拟合训练数据的学生模型要好。

自回归？自编码？

