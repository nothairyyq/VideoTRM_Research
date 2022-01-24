# VTN

CNN+LSTM中把LSTM换成Transformer

1. 

## 1. 要解决什么问题

- Transformer 用到视觉中的主要问题就是，如何构建输入序列。
  - VIT 将 Transformer 引入图像分类，就是将图像分为若干个不重叠的patch作为输入序列。
  - 那么一个视频，要如何转换为序列，作为Transformer的输入呢

对比其他算法，本文的算法能够在同样的epoch训练下获得最高的精度，并且训练的速度比最新的快16.1倍。

2. 

2. 用了什么方法
总体结构如下图所示，一共可以分为三个部分：
2D spatial feature extraction：提取每一帧的图像，可以用CNN也可以用VIT等纯transformer。
这个没啥好具体说的，其实就是提取特征，可以随机初始化并和Longformer一起训练，也可以使用预训练模型并固定权重。



3. 

temporal-base encoder：使用了 Longformer 结构

Transformer本来就是处理序列任务的，那么长序列任务有什么方法呢？之前NLP领域就提出了TimeFormer 的方法，其核心如下图所示
普通Transformer有多个q/k/v，每个q要跟所有k计算相似度+softmax、与v相乘累加，如下图左一
但随着序列长度增加，这个计算量太高了，所以每个q不是跟每个k计算相似度，而是跟部分k，具体选择哪些k就有下图的右边三张。

4. 

- classification MLP head：Longformer 的结果其实也是一个向量，最终需要通过FC进行分类。
  - 没啥好说的。

