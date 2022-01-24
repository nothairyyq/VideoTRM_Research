1. 

在CV中使用transformer，首先比较下视频理解任务和NLP的相同点：

- **Sequential 连续性：**视频和句子基本上都是连续的。
- **Contextual 具有上下文联系：**句子中某个单词的意思通常需要通过将其与句子中的其他单词联系起来来理解；对于视频来说，为了消除歧义，片段中的行为也需要与视频的其余部分结合起来。

所以，NLP的自注意模型可能会对视频建模有效。因为其不仅可以捕捉**跨时序的依赖关系**，还可以通过对**不同空间位置**的特征进行两两比较，从而揭示每一帧中的上下文信息。

2. 

在CV领域，2D/3D卷积依然是核心方法。文章也在思考，能不能使transformer完全替代卷积。

conv可操作的范围很有限，也就是它只能在自己的“价值观”范畴内行动（比如局部连接性和平移不变性）。而transformer并非如此，如果用transformer取代conv，对数据的操作就有更多的可能性了。

**Short Range Temporal-spatial Imformation 捕捉的信息范围小**
 卷积核是专门设计来捕捉**短期**时空信息的，它们不能对超出接受域的依赖关系进行建模。虽然加深网络会扩大感受野，可以从一定程度上解决这个问题，但是如果要把小范围的信息聚集得到大范围信息，这种策略会存在局限性。与之相反，Transformer中的自注意力机制通过直接比较在所有时空位置上的特征，可以被用来捕捉局部和全局的长范围内的依赖。

**耗时**
 当应用于高清的长视频时，训练深度 CNN 网络非常耗费计算资源。目前有工作（[Image Recognition1](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2010.11929)、[Image Recognition2](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2004.13621)、[Object Detection](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2005.12872)）证明，在图像领域，Transformer 训练和推导要比 CNN 更快。使得能够使用相同的计算资源来训练拟合能力更强的网络。

ViT中是将NLP中的词更换为图片中的patch，但是视频中存在大量的patch，计算量巨大，而且忽略了视频中的时空信息。为了解决这些问题，文章提出了几种基于时空容量（space-time volume）的可扩展自我注意设计结构。这其中最好的设计是“分散注意力(divided attention)”架构，它分别在网络的每个区块内应用时间注意力和空间注意力。

3. 

##### 2）方法

模型输入为![\mathbb{R}^{H×W×3×F}](https://math.jianshu.com/math?formula=%5Cmathbb%7BR%7D%5E%7BH%C3%97W%C3%973%C3%97F%7D)，代表![F](https://math.jianshu.com/math?formula=F)帧的RGB图像，每张图高![H](https://math.jianshu.com/math?formula=H)，宽![W](https://math.jianshu.com/math?formula=W)。然后将每一帧分解为![N](https://math.jianshu.com/math?formula=N)个不重叠的小块(patch)，每个小块的大小为![P × P](https://math.jianshu.com/math?formula=P%20%C3%97%20P)。和NLP 中的词向量embedding相似，我们先对每个patch线性embedding，其中使用到的位置编码信息记为![e^{pos}(p,t)](https://math.jianshu.com/math?formula=e%5E%7Bpos%7D(p%2Ct))，位置编码的具体细节还需要看源码理解，文章中没有仔细讲解。我们将embedding后的向量记为![z^{(0)}_{p,t}](https://math.jianshu.com/math?formula=z%5E%7B(0)%7D_%7Bp%2Ct%7D)，其中![E](https://math.jianshu.com/math?formula=E)表示一个可学习的矩阵：

x表示每个patch压成的向量，与一个可学习的矩阵E相乘，加上位置编码信息，得到embedding后的向量z

4. 

得到这个embedding后的向量![z](https://math.jianshu.com/math?formula=z)后，我们就可以开始transformer的主要流程了。介绍流程之前，先把**每个encoder中的结构**晒出来，以便后续的讲解

我们把每个encoder中的计算过程分成两部分，分别是**Attention**部分和**MLP**部分。



- Attention值的计算
   在第![l](https://math.jianshu.com/math?formula=l)个encoder中，编码器已经得到了上一个encoder传入的![z^{(l-1)}](https://math.jianshu.com/math?formula=z%5E%7B(l-1)%7D)。我们首先使用![z^{(l-1)}](https://math.jianshu.com/math?formula=z%5E%7B(l-1)%7D)来计算transformer中最重要的三个量——q,k,v，其中LN操作由LayerNorm完成，字面意思应该是层归一化。

得到三个重要的虚拟值后，下一步就是搬出softmax公式，来计算attention值啦

公式中的![k](https://math.jianshu.com/math?formula=k)看起来特别复杂吧，开始笔者也没理解这到底啥意思。后来定睛一看，因为**视频数据包含时、空两个维度的信息**，TimeSformer把这两个维度的attention值分开运算，所以![k](https://math.jianshu.com/math?formula=k)根据不同需要取不同的值。这个公式表示的是**Joint Space-time Att.**。

5. 

softmax操作结束后，最后一步——把得到的attention值![\alpha](https://math.jianshu.com/math?formula=%5Calpha)和value值相乘求和，得到**当前patch**和**相邻空间/时间上patch**的关联信息![s](https://math.jianshu.com/math?formula=s)

最后，将attention模块中multi-head的部分做处理，把这些单个的attention结构得到的![s](https://math.jianshu.com/math?formula=s)值拼接到一起。然后乘上权重![W_O](https://math.jianshu.com/math?formula=W_O)，与第![l-1](https://math.jianshu.com/math?formula=l-1)个编码器输出的![z^{(l-1)}](https://math.jianshu.com/math?formula=z%5E%7B(l-1)%7D)相加，实现short-cut的操作。

6. 

下面开始计算MLP部分（也就是上图中黄线以下的部分）。这部分比较简单，通过感知机嵌套LN计算得到的值，再和attention部分得到的![z'](https://math.jianshu.com/math?formula=z%27)相加，得到最后的输出值![z^{(l)}](https://math.jianshu.com/math?formula=z%5E%7B(l)%7D)。

最后收尾步骤就是完成分类任务。使用一个hidden layer的感知机，输出视频类别。

7. 

#### 3）结构设计

如前所述，我们可以将**时空注意力Joint Space-time Att.**替换为**每帧内的空间注意力**，从而降低计算成本。但是这种结构忽略了时间依赖性。正如我们的实验所示，与全时空注意力相比，这种方法导致分类精度下降，特别是在需要强时间建模的基准上。

8.  

 所以文章提出了另一种更有效的时空注意力架构，名为**“Divided  Space-Time Attention”**(用T+S表示)，将时间注意和空间注意分别应用。

**“Divided  Space-Time Attention”**先计算每个patch上的时序attention值![\alpha^{time}](https://math.jianshu.com/math?formula=%5Calpha%5E%7Btime%7D)，再计算每帧的空间attention值![\alpha^{space}](https://math.jianshu.com/math?formula=%5Calpha%5E%7Bspace%7D)。



除此之外，还提出了两种其他的结构，Sparse Local Global”(L+G) 和 “Axial”(T+W+H)。结构上不赘述了，每种结构的机制还是通过下图来理解。

空间注意力机制（S)：只取同一帧内的图像块进行自注意力机制；

时空共同注意力机制（ST）：取所有帧中的所有图像块进行注意力机制；

分开的时空注意力机制（T+S）：先对不同帧中，相同位置的patch进行注意力机制，再对同一帧中的所有图像块进行自注意力机制；

稀疏局部全局注意力机制（L+G）：先利用所有帧中，相邻的 H/2 和 W/2 的图像块计算局部的注意力，然后在空间上，使用2个图像块的步长，在整个序列中计算自注意力机制，这个可以看做全局的时空注意力更快的近似；

轴向的注意力机制（T+W+H）：先在时间维度上进行自注意力机制，然后在纵坐标相同的图像块上进行自注意力机制，最后在横坐标相同的图像块上进行自注意力机制。



9. 

对于 K400 数据集，**仅使用空间信息已经能够分类比较好了**，这些前面的研究者也发现了，但是，**对于 SSv2 数据集来说，这个数据集中的视频对时序上信息比较看重，所以仅仅使用空间信息的效果非常差**。这说明了对时间建模的重要性。

10. 

 计算量比较 .无论是在空间上还是时间上，都显著节省了计算量。

增加帧数/增加patch个数对结果的影响

在空间上，增加到一定数量，精度会下降；而时序上，增加输入帧的数量，精度持续增加。
 这里由于显存的限制，没有办法测试 96 帧以上的视频片段。作者说，这已经是一个很大的提升了， 因为目前的卷积模型，输入一般都被限制在 8-32 帧。



SSv2和Diving48上的结果，SSv2并没有达到最好的结果，作者提到说所提方法采用了完全不同的结构，对于这么有挑战性的数据集来说已经是比较好的了，有进一步发展的空间。