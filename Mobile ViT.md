# Mobile ViT

CNN: 具有局部连接和权值共享的特性，因此参数量比较少

但是CNN网络在spatial上是局部的，而基于self-attention的ViT能够学习到全局的特征表示，ViT的缺点就是参数量相对于CNN太大了

本文即结合CNN和ViT的优点设计light的ViT， MobileViT

|      |  Drawback    |  advantage    |
| ---- | ---- | ---- |
|  CNN    |  spatially local    |  spatial inductive biases - fewer parameters    |
|  ViT    | Heavy-weight     |  global representations    |
|  MobileViT    |      | light-weight ViT |

MobileNetv3 more than DeIT 3% accurate

This paper not focuse on optimizing for FLOPs. 

Focuse on light-weight, general-purpose  and low latency network



MobileViT combines the benefits of CNNs(spatial inductive biases and less sensitivity to data augmentation) and ViTs(input-adaptive weighting and global processing)

## Method

### Standard ViT

![image-20220214224332932](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220214224332932.png)

将input image **X[H,W,C]** patchhing 后进行Flatten展平 **X`[N,PC]**，后经过一个Linear层将维度缩放到d维度  **X''[N,d]**，在添加位置编码后通过L个Transformer Block学习到不同特征，最后经过Linear层输出预测。

ViT忽略了inductive biases因此parameter会更多

### Mobile ViT Block

![image-20220215112525015](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220215112525015.png)

Input tensor **X[H,W,C]**. 

- 将Input tensor使用point-wise conv- + n*n conv- 放缩channel数为d，得到**X'[H,W,d]**

  > a **n x n** standard convolutional layere encodes local spatial information

  > a **point-wise(1x1) convolutional layer** projects the tensor to a high-dimensional space

- 然后将**X'[H,W,d]** unflod为**X''[P,N,d]**. 输入transformer提取global spatial information，输出**Y''[P,N,d]** 在flod成**Y'[H,W,d]**

  > P=$ h\times w$ ,, $ N = \frac{H\times W}{P}$ the number of patches.  每个patch[P,1,d]都有h*w个像素点

- 对Y'[H,W,d] 进行PW conv-操作复原成 [H,W,C]维度，并于最开始的**X[H,W,C]** 进行拼接操作。最后使用n*n conv- 做channel融合得到最终的Y

![image-20220215124531059](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220215124531059.png)

在黑色cell中蓝色pixel通过conv-获得周围灰色pixel的local information. 红色pixel与蓝色pixel做self-attention。相当于红色pixel can encode information from all pixels in input tensor. overall effective receptive field is H*W

### Transformer replace 

Standarard convolutions: unfolding-> matrix multiplication(local representations) -> folding

MobileViT block using transformer replace matrix multiplication to learn global representations.  And havs convolution-like properties(spatial bias)



### Why light-weight?

Standard ViT convert the spatial information into latent by learning a linear combination of pixels. Global information is encoded by learning inter-patch information using transformer. ViT lose image-specific inductive bias. They require more capacity to learn visual representations. Network is deep and wide.

Mobile ViT uses conv- + transformer. design shallow and narrow models.

|      |   L   |   d   |
| ---- | ---- | ---- |
|  DeIT    | 12 | 192 |
|  MobileViT    | {2,4,3} | {96,120,144} |

MobileViT network is faster 1.85x, smaller 2x than DeIT



### MobileViT  architecure

![image-20220215152933492](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220215152933492.png)

1. 将input image 进行 standard conv-3x3 operation，并做down-sampling 2X
2. 4个MobileNetV2 block， 其中做两次2倍下采样
3. 间隔的加入mobilevit block和mobilenetv2 block
4. 使用conv-1x1进行channel compress
5. 进行global pool



使用MobileViT block 实现light vision transformer

- CNN+Transformer 使用CNN提取局部特征，使用transformer提取全局特征。感受野范围为H*W,全局都是感受野
- 

MobileViT的设计好处？

1. 保留patch的顺序和每个patch每个像素点的顺序

   不同于ViT[N,d]丧失了每个像素点的空间顺序，mobilevit增加了一维[P,N,d],能够保留每个patch中每个像素点的位置；P=WH;

   > VIT不是加了positional embedding吗，为什么说丧失了每个像素点的空间信息呢？

3. 让transformer具有卷积的特性

   标准的卷积：1. unfolding；2. Matrix multipulation 3. folding
   
   标准的卷积第二步步只能学习到局部特征，mobilevit将其替换成transformer用于获得全局特征