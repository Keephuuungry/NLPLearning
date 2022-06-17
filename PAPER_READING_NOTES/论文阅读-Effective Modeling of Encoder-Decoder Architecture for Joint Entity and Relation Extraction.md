## Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction

> 2020-AAAI
> Tapas Nayak, Hwee Tou Ng
> Department of Computer Science , National University of Singapore

#### 1 Introduction

面对的问题：实体关系联合抽取+实体重叠+Multi-words entity

解决问题的办法：提出新的representation scheme+提出encoder-decoder模型

**Contributions**

- "We propose a new representation scheme for relation tuples such that an encoder-decoder model, which extracts one word at each time step, can still find multiple tuples with overlapping entities and tuples with multi-token entities from sentences. We also propose a masking-based copy mechanism to extract the entities from the source sentence only."
- "We propose a modification in the decoding framework with pointer networks to make the encoder-decoder model more suitable for this task. At every time step, this decoder extracts an entire relation tuple, not just a word. This new decoding framework helps in speeding up the training process and uses less resources (GPU memory). This will be an important factor when we move from sentence-level tuple extraction to document-level extraction."
- "Experiments on the NYT datasets show that our approaches outperform all the previous state-of-the-art models significantly and set a new benchmark on these datasets."

#### 2 Encoder-Decoder Architecture

对关系三元组，作者使用";"作为元素之间的分隔符。对多个三元组之间，作者使用"|"作为分隔符。引入特殊符号"SOS": start-of-target-sequence token; "EOS": end-of-target-sequence token; "UNK": unknown word token;

##### 2.1 Embedding Layer & Encoder

Word Embedding由两部分组成：(1) pre-trained word vectors; (2) character embedding-based feature vectors。其中(2): Following Chiu and Nichols (2016), we use a convolutional neural network with max-pooling to extract a feature vector for every word. 

##### 2.2 Word-level  Decoder & Copy Mechanism

