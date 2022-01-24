# TimeSFormer

[原文连接]([Is Space-Time Attention All You Need for Video Understanding.pdf](file:///C:/Users/86133/Desktop/video%20trm/paper/Is%20Space-Time%20Attention%20All%20You%20Need%20for%20Video%20Understanding.pdf))

这篇文章的方法基于Transformer提出了一种用于video understanding的框架： - Timesformer



## Transformer Review

#### Encoder
Encoder由 **self-attention**和 **Feed Forward NN**组成

#### Decoder
decoder除了和encoder相同的两部分外，在注意力模块和FFN中间插入了`Encoder-Decoder Attention`模块

1. self-attention: 当前翻译和已经翻译的前文之间的关系
2. encoder-decoder attention：当前翻译和编码的特征向量之间的关系

![Encoder, Decoder](https://upload-images.jianshu.io/upload_images/16637214-e237e81302124d89.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

## Transformer的运作方式
### 1. Embedding

embedding指将单词变成可以用实数表示的词向量
1. 某种程度上，就是用来降维的，降维原理是矩阵乘法
2. 对低维的数据进行升维时，把一些特征放大了，或者笼统的将特征分开了
	

Embedding作为桥梁的存在，将手头的东西变得可以伸缩成我们想要的
一般采用Word2Vec，glove预训练将input embedding

### 2.  Encoding-self attention

3. Encoding-FNN
4. Decoding

# Motivation

## 1） cv VS nlp

video understanding 和 NLP的相同点：

- Sequential：video和句子都是连续的
- contextual: 具有上下文联系，句子中的某个单词的意思通常需要通过与句子中其他单词联合起来理解。对于video而言，为了消除歧义，片段中的行为也需要与视频的其余部分结合起来

通过共同点可以得出NLP的self-attention结构会对video understanding有效：可以捕捉**跨时序的依赖关系**，还可以对**不同空间位置**的特征进行相互比较，揭示video中每一帧的上下文关系

## 2） Transformer为什么可以替代卷积？

卷积操作存在的一定缺陷：

1. **Inductive Bias 归纳偏置**

   conv操作的固有价值观：conv认为这样操作可以得到有用的特征。

   在归纳偏置下，conv可操作的范围很有限，就只是conv只能在自己的价值观范畴内行动。

   而transformer可以对数据的操作有更多的可能性

2. **Short Range Temporal-spatial Imformation 捕捉的信息范围小**

   卷积核是专门设计来捕捉**短期**时空信息的，它们不能对超出接受域的依赖关系进行建模。虽然加深网络会扩大感受野，可以从一定程度上解决这个问题，但是如果要把小范围的信息聚集得到大范围信息，这种策略会存在局限性。与之相反，Transformer中的自注意力机制通过直接比较在所有时空位置上的特征，可以被用来捕捉局部和全局的长范围内的依赖。

3. **耗时**

   [Image Recognition1](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2010.11929)、[Image Recognition2](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2004.13621)、[Object Detection](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2005.12872)）证明，在图像领域，Transformer 训练和推导要比 CNN 更快。使得能够使用相同的计算资源来训练拟合能力更强的网络。

# TimeSformer

## 1) Background

用于图像的Transformer [ViT](https://arxiv.org/abs/2010.11929)的拓展

ViT将NLP中的词换成了图片中的patch。视频中的patch太多了，计算量太大，并且ViT没有考虑视频的时空因素。

这篇文章提出了几种基于`space-time volume`的结构，其中最好的是divided attention架构，其分别用在网络的每个block内应用时间和空间attention

该模型通过将**每个图像块的语义与视频中的其它图像块进行比较，来获取每个图像块的语义**，从而可以同时捕获到邻近的图像块之间的局部依赖关系，以及远距离图像块的全局依赖性。

