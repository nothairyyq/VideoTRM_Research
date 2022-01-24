# Video Swin Transformer

个人以为Swin Transformer最大的特点是类似于cnn中conv + pooling的结构。在Swin Transformer中，这种结构变成了Swin Transformer Block + Patch merging，通过多个stage，token数越来越少，每个token的感受野也会越来多大，同时由于token数的递减以及特殊设计的Window Transformer计算方式，减少了模型的计算量，可以说Swin Transformer是效果和速度双丰收。

1. 

由于Transformer强大的建模能力，视觉任务的主流Backbone逐渐从CNN变成了Transformer，其中纯Transformer的结构也在各个视频任务的数据集上也达到了SOTA的性能。这些视频模型都是基于Transformer结构来捕获patch之间全局的时间和空间维度上的关系

在本文中，作者提出了video Transformer中的局部性假设偏置，这能使Transformer在速度和精度上达到更好的trade-off，这在以前的那些基于捕获时空域上全局关系的Transformer上是做不到的.视频结构中的局部性是通过Swin Transformer实现的。

2. 

正在进行从卷积神经网络(CNN)到Transformer的转变。这一趋势始于Vision Transformer(ViT)的引入，ViT成功之处主要在于捕获了不重叠Patch之间的全局关系

对于以前的卷积模型，视频任务的Backbone主要就是增加了一个卷积维度用于捕获时间上的关系。由于联合时空（时间-空间）建模比较费计算资源并且不容易优化.   分解时空建模，来达到更好的速度-精度权衡。在Transformer中，也有类似的工作，同样起到了比较好的速度-精度权衡作用。

在本文中，作者提出了一种基于Transformer的视频识别主干网络结构，并且它在效率上超过了以前的分解时空建模的模型。因为视频数据在时间和空间上存在局部性（也就是说：**在时空距离上更接近的像素更有可能相关** ），所以作者在网络结构中利用了这个假设偏置，所以达到了更高的建模效率。

作者通过Swin Transformer[1]来实现这一点，因为Swin Transformer也考虑了空间局部性、层次结构和平移等变性等假设偏置。

3. 

严格遵循原始Swin Transformer的层次结构，但将局部注意力计算的范围从空间域扩展到时空域。由于局部注意力是在非重叠窗口上计算的，因此原始Swin Transformer的滑动窗口机制也被重新定义了，以适应时间和空间两个域的信息

	4. 

Video Swin Transformer 也是有三个部分组成，段：video to token， model stages，head。

**Video to token**

类似于VIT，直接将输入图像划分为互不重叠的多个图像块，然后利用线性层对图像块进行embedding，再加上position embedding。在Swin Transformer中，每个图像块的大小是4 x 4。

在image to token中，是将4x4的图像块作为一组，而在Video to token中，将2 X 4 X 4 的视频块作为一组，而后再进行线性embedding以及position embedding。

**Model stages**

Model stages由多个重复的stage组成，每个model stages包括Video Swin Transformer Block 和 Patch merging组成

1）Video Swin Transformer Block 由可以分为两部分，Video W-MSA 和 Video SW-MSA。MSA即为transformer 中的MSA（Multihead self attention），这里不过是加了window和shift window

这里相当于将Swin Transformer Block计算由二维拓展到三维。

2）Patch merging类似于max pooing，将相邻（2X2窗口内）token特征合并，而后再利用线性层降维，相当于将token减少4倍，不过降维并没有保持维度不变，而是每次Patch merging 之后特征维度仍然会增加2倍，和cnn中特征图减少，通道数增加很类似，这里每次进行patching merging，视频帧数是不变的。

**head**

在经过Model stages之后，得到了多帧数据的高维特征，用于视频分类的话需要进行简单的帧融合（average），作者代码中用的是I3DHead。

5. 

与图像相比，视频需要更多的输入token来表示它们，因为视频另外有一个时间维度。因此，一个全局的自注意模块将不适合视频任务，因为这将导致巨大的计算和内存成本。在这里，作者遵循Swin Transformer的方法，在自注意模块中引入了一个局部感应偏置

在每个不重叠的二维窗口上的MSA机制已被证明对图像识别是有效并且高效的。在这里，作者直接扩展了这种设计到处理视频输入中。给定一个由 个3D token组成的视频，3D窗口大小为 ，这些窗口以不重叠的方式均匀地分割视频输入。这些token被分成了多个不重叠的3D窗口。
————————————————

6. 

先前的工作已经表明，在自注意计算中包含相对位置编码对于performance的提升是有用的。因此作者在Video Swin Transformer也引入了3D**相对位置编码**

7. 

由于Video Swin Transformer改编于Swin Transformer，因此Video Swin Transformer可以用在大型图像数据集上预训练的模型参数进行初始化。与Swin Transformer相比，Video Swin Transformer中只有两个模块具有不同的形状，分别为：线性embedding层和相对位置编码。

输入token在时间维度上变成了2，因此线性embedding层的形状从Swin Transformer的48×C变为96×C。


相对位置编码矩阵的形状为 ，而原始Swin Transformer中的形状为 。为了使相对位置编码的矩阵一样，作者将原来的 相对位置编码矩阵复制了 次


8. 

作者提出了下面四种不同参数量和计算量的网络结构：