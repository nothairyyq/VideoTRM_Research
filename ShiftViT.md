# ShiftViT

## Abstract

It does not contain any parameter or arithmetic calculation

The only operation is to exchange a small portion of the channels between neighboring features.

The attention mechanism can be even replaced by a zero-parameter operation

## Introduction

The attention mechanism leverages a self-attention matrix to aggregate features from arbitrary locations

**Compared with the convolution operation in CNN **:

2 Strengths:

1. this mechanism opens a possibility to simultaneously capture both short- and long- ranged dependencies, and get rid of the local restriction of the convolution

   Swin-ViT can introduce a local attention mechanism to restrict their attention scope tithin a small local region

2. the interaction between two spatial locations dynamically depends on their own features, rather than a fixed convolutional kernel.

Transformer取得成功在于两个特性：

1. Global：快速的全局建模能力，每个token都能和其他token发生关联
2. Dynamic：为每个样本动态的学习一组权重

**The paper replace the former attention layer with a shift operation**

ViT building block consists of two parts: attention layer + feed-forward network

keeping the latter FFN part untouched. Given an input feature, the proposed building block will first shift a small portion of the channels along four spatial directions(top,down,left,right)

The information of neighboring features is explicitly mingled by the shifted channels

## Shift Operation Meets Vision Transformer

The modules can be divided into 4 stages, each stage contains two parts:

1. embedding generation

   - Linear projection layer:

     map each token into an embedding of channel size $C$

   - Patches merging

     through the convolution with a kernel size of 2*2

     After patch merging, the spatial size of the output is half down-sampled, while channel size is twice the input

   ![image-20220207165255905](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220207165255905.png)

2. stacked shift blocks

   - Shift operation

     partial shift operation in TSM

   - Layer normalization

   - MLP network

   ![image-20220207165310132](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220207165310132.png)

​	attention operation-> shift operation

### Shift operation

这个模块非常简单，就是将输入维度为CHW的特征，沿C这个方向取出来一部分，然后平均分为4份，这4份特征分别沿 左、右、上、下 进行移动，剩下部分的特征保持不变。

![image-20220207171437752](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220207171437752.png)

input tensor, a small portion of channels will be shifted along 4 spatial directions, namely left, right, top, and down, while the remaining channels keep unchanged



![image-20220207165821550](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220207165821550.png)

shift operation does not hold any parameter or arithmetic calculation(only memory copying)

在作者的实现中，shift的步长设置为1个像素，同时，选择 1/3 的通道进行 shift （1/12的通道左移1个像素，1/12的通道右移1个像素，1/12的通道上移1个像素，1/12的通道下移1个像素）。该模块的 pytroch代码如下，可以看出来，这个模块计算非常简单，基本没有参数。

![image-20220207170926247](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220207170926247.png)

