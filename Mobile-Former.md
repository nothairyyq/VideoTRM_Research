# Mobile-Former

Parallel Design: MobileNet and Transformer (2 bridge)

MobileNet at local processing and transformer at global interaction

将mobilenet和transformer并行连接设计 

这样的结构可以将mobilenet的local information和transformer的global information进行一个结合

用双向连接将local feature和global feature进行特征融合

这个model只有6个token

主要优点是computationally efficient 和more representation

![image-20220307165114460](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307165114460.png)

![image-20220307165127443](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307165127443.png)

## Introduction

ViT虽然性能好但是在1G FLOPs的限制下，ViT的性能就不好了

MobileNet: decomposition of depthwise and pointwise conv-

### Motivation:

How to design efficient networks to effectively encode both local processing and global interaction? 

MobileViT是将CNN和Transformer串联在一起的，FLOPS太大了

B,192,768 -> B,6,192

![image-20220307171507250](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307171507250.png)

Left: Mobile Net, 
takes an image as input and stacks mobile blocks
It use depthwise and pointwise convolution to extract local 							    features

Right: Former 
takes a few learnable tokens as input(6)
stacks multi-head attention and FFN.  global features of image



Communication:

two-way bridge to fuse local and global features

It feeds local features to Former's tokens  And introduces global views to every pixel of feature map

This bidirectional bridge by performing the cross attention at the bottleneck of Mobile where the channel number is low

在bottleneck处进行attention计算

在通道数比较少的时候建立bridge可以有效减少因为特征融合所失去的信息.融合更稳定，不会丢失信息

Mobile-Former block

![image-20220307172930409](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307172930409.png)

Removing projection on kqv from mobile side

input: local, feature map

1. 左侧$X_i$作为**输入**，是一组`Feature Map`; 右侧$Z_i$作为**输入**，是一组`Token`.
2. 左侧的输入$X_i$先传入右侧的**黄色区域**，将特征图中的局部信息融入到$Z_i$中——这一步通过注意力映射来完成，但是仅对输入的Token进行`Query`的映射.
3. 上面的这一步，不对特征图进行映射，而是直接当作`Key`和`Value`进行注意力计算，最后利用残差将局部信息融入到Token中的全局信息里——称为`Mobile->Former`，.
4. 通过桥接后，将补充信息后的$Z_i$传入`Former结构`，即**绿色区域**——一个单纯的Transformer结构，完成特征提取后输出$Z_{i+1}$.
5. 拿到$Z_{i+1}$后，将其作为当前阶段完整的全局信息Token，传回到`Mobile`结构，即**蓝色区域**, 此时才将$X_i$传入`Mobile`结构——实现全局信息补充到局部信息的特征融合/交互过程.
6. 这是信息的交互，主要是通过Token提供给动态ReLU函数生成动态参数，去对卷积过程的局部信息/特征进行一个筛选的交互过程.
7. $X_i$经过Mobile结构后，信息并没有真正的补足全局的认识，因此，还需要与表示全局信息的Token进行一次注意力计算.
8. 此时再将$Z_{i+1}$经过两次映射，得到Key和Value，而经过Mobile输出的$X_i$保持原有特征，直接作为Query指导全局信息的融入，最后通过残差进行特征融合——称为`Former->Mobile`.
9. 这样一个交互过程，由于特征图始终没有进行任何映射，保持原汁原味的局部信息——更加纯净的融入到Token中，并且Token中表示全局的信息也在多个这样的block中不断的更新并融合到特征图的特征上.

a light-weight cross attention to model them, in which the projections (WQ, WK, WV) are removed from Mobile side to save computations, but kept at
Former side.

![image-20220307205947673](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307205947673.png)

X: local feature map

Z: global tokens

$W^{O}$ is used to combine multiple heads together

![image-20220307210221496](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220307210221496.png)
