# Parameter Efficient Multimodal Transformers For Video Representation Learning

## Abstract

Focus on :

- reducing the parameters of multimodal Transformers in the context of audio-visual video representation learning.

- alleviate the high memory requirement
  - share the parameters of Transformers across layers and modalities (based on low-rank approximation)
- Negative sampling approach based on an instance similarity measured on the CNN embedding space 

## Introduction

Key Contributions

### 1. End-to-end trained multimodal Transformer

An end-to-end trainable bidrectional transformer architecture that learns contextualized audio-visual representations of long videos.

- CNNs operate on short video clips and are intended to capture short-term dynamics within each modality.

- Transformer layers operate on long video sequences capturing long-term dynamics

Parameter reduction technique that shares parts of weight parameters across Transformers and across layers within each Transformer

![image-20220126222243429](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220126222243429.png)

### 2. negative samples

Content-aware negative sampling strategy that favors negatives sufficiently similar to a positive instance.

### 3. systematic evaluation of different modality fusion strategies

we compare three fusion strategies (early, mid, late) and show the superiority of mid-level fusion.

## Approach

#### Negative Sampling

For the model's convergence

a content-aware negative sampling strategy that favors negatives sufficiently similar to a positive instance in the CNN embedding space;

Ulyanov et al. (2018) who showed that randomly initialized CNNs provide a strong prior over natural images due to the inductive bias already built into the design of the CNNs

can be sufficient to assess the similarity/dissimilarity between clips.

> idea:这种方法是不是对于ssv2动作检测的任务也很有用？因为可以更好的捕捉到clips之间的信息，有更多temporal上的信息？
>
> 而且ALBERT也是共享权重+SOP的方法轻化bert

1. compute $l_{2}$ and $x_{t}$(positive) distance, then normalize them to the[0,1] interval

2. remove samples(too similar/different) from positive sample, and outside the 95% confidence interval in normalized space
3. using normalized distance as sampling probability: sample the negatives from the remainder

#### Parameter Reduction

##### Sharing across Transformers

Each layer weights: ${W^{q},W^{k},W^{v},W^{b},W^{c},W^{d}}$

Decompose each of these weights into: $W=U\Sigma V^{T}$ $W ∈ R^{M×N} , U ∈ R^{M×O}, Σ ∈ R^{O×O}, V ∈ R^{N×O}.$

Setting the rank O<<M,N

U: shared across Transformers; $\Sigma V$: private to each transformer

Result:

MO+ 3(O2 +NO) << 3MN

![image-20220126233748566](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220126233748566.png)

##### Sharing across Layers

ALBERT的内容



> Idea: 上述的方法我觉得都可以重复使用在新的ViT模型上，如MVIT,MaskFeat这些
>
> 除此之外是不是可以考虑除了sharing parameter之外的方法，例如pruning parameter

![preview](https://pic2.zhimg.com/v2-a142cf6b751e5ad987d0997d27413415_r.jpg)



# Pruning Self-attention

**https://arxiv.org/abs/2111.11802**

现有的Transformer剪枝技术大致可以分为module剪枝和token剪枝。

module剪枝对不显著的Transformer modules进行修剪，如MSA层中的heads、linear projections通道和权重神经元。

最近，有学者提出动态识别和修剪不太重要的token特征维度。Chen等人在lottery ticket假设下提出了Transformer参数的prune-and-grow方法。

token剪枝侧重于修剪不太重要的token，如DynamicViT分层地修剪冗余token，从而在分类任务中实现FLOPs减少。然而，随着一些token的消除，将token剪枝方法应用于密集预测任务是一个挑战，如分割。

本文工作旨在对modules进行剪枝，但与之前的工作不同的是，本文着重于对MSA层中多余的全局关联进行剪枝。并没有以多路径的方式将卷积和MSA独立地添加到搜索空间中，而是考虑了这两种操作之间的内在关系，并将MSA层剪成卷积层，这允许以低搜索成本快速部署高效的Transformer模型。



