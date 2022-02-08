# TerViT

## Abstract

A ternary vision transformer to ternarize the weights in ViTs.

Challenge is the large loss surface gap between real-valued and ternary parameters. :

A scheme: first training 8-bit transformers and the TerViT

## Introduction

TerViT: improve the loss landscape of TerViT for a better optimization. Take the 8-bit model as a proxy to bridge the gap between the ternary and real-valued models

ternary weights converts the latent full-precision weights into a proxy 8-bit model to initialize the training of TerViT

2 contributions:

1. a ternary vision transformer for the quantization of vision transformer
2. a channel-wise quantization scheme, which can improve the quantization stability without increasing the model complexity.

## Method

### Quantization for vision transformers

Directly quantizing each 3 matrices in MHSA as an entirety with the same quantization range can significantly degrade the accuracy

**Channel-wise ternarization**

![image-20220127135147704](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220127135147704.png)

$\bigcirc $ denotes the channel-wise multiplication.

$\alpha $  is the channel-wise scale factor defined by the channel-wise absolute mean as

![image-20220127135618814](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220127135618814.png)

![image-20220127135704014](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220127135704014.png)

### Activation quantization

two kinds of commonly used 8-bit quantization methods:\

- symmetric 8-bit quantization

  the quantized values of the symmetric 8-bit quantization distribute symmetrically in both sides of 0

- min-max 8-bit quantization

  min-max 8-bit quantization distribute uniformly in a range determined by the minimum and maximum values

![image-20220127141954731](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220127141954731.png)

