# Transformer Quality in Linear Time



Transformer's weaknesses in handling long sequence.

1. Gated attention unit: the use of a weaker single-head attention with minimal qulity loss 提出了一种新的Transformer变体，它依然具有二次的复杂度，但是相比标准的Transformer，它有着更快的速度、更低的显存占用以及更好的效果；

2. Linear approximation: complementary with gated attention unit, and accelerator-friendly and highly competitive in quality. Mixed chunk attention

   提出一种新的线性化Transformer方案，它不但提升了原有线性Attention的效果，还保持了做Decoder的可能性，并且做Decoder时还能保持高效的训练并行性 

Existing efficient-attention method have at least one of the following drawbacks:

1. Inferior Quality:

   efficient Transformer实验效果的大幅度滑落，这种滑落已经明显盖过了 efficiency 所带来的增益

2. Overhead in Practice:

   现有的 efficient Transformer 虽然只需要更少的参数个数，但是在网络结构上将 Transformer layer 复杂化了，一方面缺乏可解释性，另一方面难以填平该结构的理论复杂度与GPU、TPU的现实算力之间的巨大差距

3. Inefficient Auto-regressive Traning

   这样类似于 RNN 的网络结构在 decoding时十分高效，但是在 auto-regressive training 的时候却是非常缓慢

### optimization：

1. 目前的这些 efficient Transformer 都是以 multi-head self-attention (MHSA) 作为核心点的. FLASH(Fast Linear Attention with a Single Head) 把原先的 gated linear unit (GLU) + Multi-Head Self-Attention 的网络结构改成了 Gated Attention Unit

   Gated Attention Unit. GAU layer is cheaper, and its quality relies less on the precision of attention. GAU with a small single-head虽然仍然有着quadratic 复杂度，但是不再依赖于attention 因此可以minimal quality loss 来近似

​				GAU 替代 self-attention 和 FFN![image-20220308120017976](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220308120017976.png)

​		

2. Layer variant with linear complexity over the complexity over the context size:

   group tokens into chunks, then using precise quadratic attention within a chunk and fast linear attention across chunks

![image-20220308121052076](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220308121052076.png)



## GAU

标准FFN:

两层MLP模型： $O = \phi (XW_{u}) W_{o}$

 $X \in \mathbb{R}^{T\times d}$    $ W_{u} \in \mathbb{R}^{d\times e}$    $ W_{o} \in \mathbb{R}^{e\times d}$   

$\phi$: activation function.  T: tokens. d: model size. e: expanded intermediate size



使用GLU的FFN效果更好：

$U = \phi (XW_{u})$, $V=\phi(XW_{v})$, $O=(U\bigodot V) W_{o}$

$\bigodot$: element-wise multiplication 这篇paper中UV都加了Swish激活函数，和普通glu不一样



GAU:

GLU中各个token之间没有进行交互（$u_{i},v_{i}$每一行独立运算）。GAU中把token之间的联系补充到U,V上去

> 这种方式应该也能放在VIT上？

把token之间的联系补充到UV上，

$O = (U\bigodot \hat{V})W_{0}$,  $\hat{V}=AV$  A是token-token的attention权重矩阵，负责融合token之间的信息。因此O就包含了token之间的交互，因此可以取代attention

如果`A==I`: 上式就是GLU的FFN, 如果U是全1矩阵，那么就是普通的self-attention

所以GAU是一个attention和ffn的融合，希望同时替换掉attention和ffn



计算A

$Z=\phi_{z}(XW_{z})   \in \mathbb{R^{T\times s}}$  这里的s和上面UV的维度e不一样，可以设置的很小以节省参数量 `s:head_size s=128`

$A={relu}^{2}(Q(Z)K(Z)^{T}+b) \in \mathbb{R}^{T\times T}$

RELU平方代替softmax作为激活函数

![image-20220308154206778](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220308154206778.png)



## Signgle head

GLU对于attention的依赖比较低，因此做实验发现single-head就可以产生很好的效果

参数量：

普通attention：MHA:4d^2, FFN:8d^2  ->> 12d^2

GAU: Z的参数为ds，s设置的很小，因此只有3de

当e=2d时，1 attention+FFN = 2GAU



## Mixed Chunk Attention

不仅可以应用在GAU中，attention中也可以用(performer,cosFormer,FlowFormer)

实现：

标准attention： $\phi (QK^T)V$

linear attention: $ (\phi _{q}(Q) \phi_{k}(K)^{T}) V$

问题：low-rank会导致结果变差。用来做decoder会牺牲训练的并行性，要是转换成RNN计算又需要更高的计算复杂度



FLASH采取了“局部-全局”分块混合的方式，结合了“稀疏化”和“线性化”的优点。首先，对于长度为![[公式]](https://www.zhihu.com/equation?tex=n)的输入序列，我们将它不重叠地划分为![[公式]](https://www.zhihu.com/equation?tex=n%2Fc)个长度为![[公式]](https://www.zhihu.com/equation?tex=c)的块（不失一般性，假设![[公式]](https://www.zhihu.com/equation?tex=c)能被![[公式]](https://www.zhihu.com/equation?tex=n)整除，论文取![[公式]](https://www.zhihu.com/equation?tex=c%3D256)），设![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BU%7D_g%2C%5Cboldsymbol%7BV%7D_g%5Cin%5Cmathbb%7BR%7D%5E%7Bc%5Ctimes+e%7D%2C%5Cboldsymbol%7BZ%7D_g%5Cin%5Cmathbb%7BR%7D%5E%7Bc%5Ctimes+s%7D)为第![[公式]](https://www.zhihu.com/equation?tex=g)块，其中![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BU%7D%2C%5Cboldsymbol%7BV%7D%2C%5Cboldsymbol%7BZ%7D)的定义同前。将![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BZ%7D_g)通过4个简单的变换分别得到![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%2C%5Cboldsymbol%7BK%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%2C%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Blin%7D%7D%2C%5Cboldsymbol%7BK%7D_g%5E%7B%5Ctext%7Blin%7D%7D)。

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%2C%5Cboldsymbol%7BK%7D_g%5E%7B%5Ctext%7Bquad%7D%7D)我们用来算块内的自注意力：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cboldsymbol%7BV%7D%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%3D%5Cfrac%7B1%7D%7Bcs%7D%5Ctext%7Brelu%7D%5E2%5Cleft%28%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%7B%5Cboldsymbol%7BK%7D_g%5E%7B%5Ctext%7Bquad%7D%7D%7D%5E%7B%5Ctop%7D%5Cright%29%5Cboldsymbol%7BV%7D_g+%5C%5C)

这代表的是每个块的token内部自行交互，本质上也算是“稀疏化”的一种，其复杂度大致是![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr%7BO%7D%28n%2Fc%5Ctimes+c%5E2%29%3D%5Cmathscr%7BO%7D%28nc%29)，正比于![[公式]](https://www.zhihu.com/equation?tex=n)。实现时相当于头数为![[公式]](https://www.zhihu.com/equation?tex=n%2Fc)、序列长度为![[公式]](https://www.zhihu.com/equation?tex=c)的多头注意力，可以充分地并行，而如果想要做Decoder，那么mask掉注意力矩阵的上三角部分即可。

剩下的![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Blin%7D%7D%2C%5Cboldsymbol%7BK%7D_g%5E%7B%5Ctext%7Blin%7D%7D)则用来做全局的Attention，直接用前述线性Attention的方式来做：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cboldsymbol%7BV%7D%7D_g%5E%7B%5Ctext%7Blin%7D%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Blin%7D%7D%5Csum_%7Bh%3D1%7D%5E%7Bn%2Fc%7D+%7B%5Cboldsymbol%7BK%7D_h%5E%7B%5Ctext%7Blin%7D%7D%7D%5E%7B%5Ctop%7D%5Cboldsymbol%7BV%7D_h+%5C%5C)

注意，这个操作跟直接用完整矩阵![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BQ%7D%5E%7B%5Ctext%7Blin%7D%7D%2C%5Cboldsymbol%7BK%7D%5E%7B%5Ctext%7Blin%7D%7D%5Cin%5Cmathbb%7BR%7D%5E%7Bn%5Ctimes+s%7D)与![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BV%7D)做线性Attention是完全等价的，写成这样只是更好地体现跟分块的联系。如果是做Decoder，那么要防止泄漏未来信息，所以要改为cumsum形式：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cboldsymbol%7BV%7D%7D_g%5E%7B%5Ctext%7Blin%7D%7D%3D%5Cfrac%7B1%7D%7B%28g-1%29n%2Fc%7D%5Cboldsymbol%7BQ%7D_g%5E%7B%5Ctext%7Blin%7D%7D%5Csum_%7Bh%3D1%7D%5E%7Bg-1%7D+%7B%5Cboldsymbol%7BK%7D_h%5E%7B%5Ctext%7Blin%7D%7D%7D%5E%7B%5Ctop%7D%5Cboldsymbol%7BV%7D_h+%5C%5C)

这种情况下，为了保持并行性，只需要![[公式]](https://www.zhihu.com/equation?tex=b%28n%2Fc%29se)的空间复杂度，而如果不分块直接用线性Attention，那么是![[公式]](https://www.zhihu.com/equation?tex=bns%5E2)（要是原始的用法还要加上多头，那就是![[公式]](https://www.zhihu.com/equation?tex=bhns%5E2)），在当前参数设置下有![[公式]](https://www.zhihu.com/equation?tex=e%2Fc%5Cll+s)，所以是更省显存了。

最后，将两种Attention结果结合起来，整合到GAU中，得到线性版本的GAU

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7BO%7D_g%3D%5Cleft%5B%5Cboldsymbol%7BU%7D_g%5Codot%5Cleft%28%5Chat%7B%5Cboldsymbol%7BV%7D%7D_g%5E%7B%5Ctext%7Bquad%7D%7D+%2B+%5Chat%7B%5Cboldsymbol%7BV%7D%7D_g%5E%7B%5Ctext%7Blin%7D%7D%5Cright%29%5Cright%5D%5Cboldsymbol%7BW%7D_o+%5C%5C)

基于线性版本GAU搭建的Transformer模型，FLASH模型





![image-20220308162013532](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220308162013532.png)

1、尽管FLASH-Quad和Transformer都是二次复杂度，但FLASH-Quad效果更好、速度更快；

2、在序列足较长时，线性复杂度的FLASH比FLASH-Quad更快，并且效果相仿。

