# Unified Visual Transformer Compression

## Abstract

ViT drawback: computational over head.

​	-  due to stacking multi-head self-attention modules and else

Propose a unified ViT compression framework:

 - including: a. pruning
    				   2. layer skipping
        3. knowledge distillation

Targeting:

	- jointly learning model weights;
	- layer-wise pruning ratios/masks
	- skip configurations

## Introduction

### NLP compression pruning method

1. unstructured pruning
2. attention head pruning
3. encoder unit pruning

### ViT compression works

1. weight/attention pruning
2. input feature(token) selection
3. knowledge distillation

### Motivation

there has been no systematic study that strives to either compare or compose multiple individual compression techniques for ViTs

This paper: the first all-in-one compression framework:

1. structured pruning
2. block skipping
3. knowledge distillation

UVC on DeiT-Tiny (with/without distillation tokens) yields around 50% FLOPs reduction, with little performance degradation (only 0.3%/0.9% loss compared to the baseline).

## Related Works

### Pruning

1. unstructured pruning

   removing insignificant weight via certain criteria

   causing sparse matrix operations that are hard to accelerate on hardware

2. structured pruning

   zero out parameters in a structured group manner

   calculate an importance score for some group of parameters (e.g., convolutional channels, or matrix rows)

transformer-based models pruning:

	- blocks
	- attention heads
	- fully-connected matrix rows

### Knowledge distillation

teacher-student model

### Skip configuration



## Method

### purning

$W_{Q}^{(l)}, W_{K}^{(l)}, W_{V}^{(l)}$: self-attention weights of the three linear projection matrix 

$W^{(l,1)},W^{(l,2)},W^{(l,3)}$: self-attention,MLP linear projection

prune the head number and head dim- inside each layer

不修建QKV的计算(since qkv should be of the same shape for computing self-attention)，innovate to use![image-20220208134937253](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208134937253.png) and ![image-20220208135144914](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208135144914.png), as the proxy pruning targets.

pruning on these linear layer is equal to the pruning head number and dim-

skip connection $W^{l,2}$, because the dimension between input and output should be aligned

![image-20220208135720357](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208135720357.png)

head dim-: $r^{(l,i)}$; the number of output neurons to be purned for each attention head i

head number level: $s^{(l,3)}$;  the number of rows to be pruned for weights w3

### Skipping manipulation across blocks

1. ViT have uniform internal structures

   all input/output size are identical

   That allows to drop any of those components and directly reconcatenating the others, without causing any feature dimension incompatibility

2. the top few blocks of fine-tuned transformers can be discarded

   That is because deeper layer features tend to have high cross-layer similarities (while being dissimilar to early layer features), making additional blocks add little to the discriminative power

denote $gt^{(g,1/2)}$ as two binary gating variables to decide whether skip this block or not

![image-20220208144702635](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220208144702635.png)

