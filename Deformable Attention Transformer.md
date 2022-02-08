# Deformable Attention Transformer

## Abstract

### Motivation

1. ViT leads to excessive memory and computational cost, 

   and features can be influenced by irrelevant parts which are beyond the region of intersts

2. sparse attention is data agnostic and may limit the ability to model long range relations

### Deformable self-attention module

the positions of K and V pairs in self-attention are selected in a data-dependent way.

Which enables the self-attention module to focus on relevant regions andcapture more informative features

## Introduction

### Motivation

the excessive number of keys to attend per query patch yields high computational cost and slow convergence, and increases the risk of overfitting



保持 Q 不变，针对每个 K / V 学习一个位置偏差，取该处的特征值即可。

<img src="C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208164842010.png" alt="image-20220208164842010" style="zoom:67%;" />

- ViT 中所有 Q 的感受野是一样的，都针对全局所有位置特征；
- Swin 中则是局部 Attention，因此处于不同窗口的两个 Q 针对的感受野区域是不一样的；
- DCN 则是针对周围九个位置学习偏差，之后采样矫正过的特征位置，可以看到图中红点蓝点数量均为 9；
- 本文提出的 DAT 则结合了 ViT 和 DCN，所有的 Q 会共享相同的感受野，但这些感受野会有学出来的位置偏差；为了降低计算复杂度，针对的特征数量也会降采样，因此图中采样点一共 16 个，相比原来缩小了 1/4。

## Deformable Attention Transformer

#### ViT

![image-20220208170645842](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208170645842.png)

![image-20220208170651926](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208170651926.png)

#### Deformable attention

![image-20220208170847972](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208170847972.png)

只对kv进行改变，操作是基于位置改变后的插值

k,v tilde represent the deformed key and value embeddings

Δp是基于q learning的offset

φ(·; ·) to a bilinear interpolation to make it differentiable:

![image-20220208171656840](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208171656840.png)

插值操作，实际上是四个角对中间取一个加权。 g(a,b)就是新点和原点之间的距离之比，normalize在0-1之间。z是满足0-1范围的点，即新点p周围的四个像素点

![image-20220208171956481](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208171956481.png)

![image-20220208172707820](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208172707820.png)

![image-20220208173651976](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208173651976.png)