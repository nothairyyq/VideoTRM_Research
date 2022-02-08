# ALBERT

## Abstract

To address  GPU/TPU memory limitations and longer training times , we present two parameter  reduction techniques to lower memory consumption and increase the training speed of BERT

## Introduction

memory limitation problem solutions(not the communication overhead): 

1. model parallelization
2. clever memory management

ALBERT 2 parameter reduction techniques:

 1. factorized embedding parameterization

    By decomposing the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden layers from the size of vocabulary embedding

    > Idea: 是否可以将vocabulary embedding 看成一个frame的image信息，将frame分成patch后，把每一个patch再按照这种方法分成 two small matrices

2. cross-layer parameter sharing

   This technique prevents the parameter from growing with the depth of the network.

## Related Works

### scaling up representation learning for NLP

using larger hidden size, more hidden layers, and more attention heads always leads to better performance

 they stop at a hidden size of 1024, presumably because of the model size and computation cost problems.

- a method called gradient checkpointing to reduce the memory requirement to be sublinear at the cost of an extra forward pass.

https://arxiv.org/abs/1604.06174

- a way to reconstruct each layer’s activations from the next layer so that they do not need to store the intermediate activations.

https://arxiv.org/abs/1707.04585

our parameter-reduction techniques reduce memory consumption and increase training speed

## ALBERT Innovation

 denote the vocabulary embedding size as E, the number of encoder layers as L,  the hidden size as H

### Factorized embedding parameerization

WordPiece embeddings are meant to learn context-independent representations

 hidden-layer embeddings are meant to learn context-dependent representations.

increasing H increases the size of the embedding matrix, which has sizeV ×E. This can easily result in a model with billions of parameters, most of which are only updated sparsely during training

use a factorization of the embedding parameters, decomposing them into two smaller matrices. Instead of projecting the one-hot vectors directly into the hidden space of size H, we first project them into a lower dimensional embedding space of size E, and then project it to the hidden space. By using this decomposition, we reduce the embedding parameters from O(V × H) to O(V × E + E × H)

首先说下BERT-base是由12层Transformer中encoder层组成，我们用BERT获得单词或句子的向量表示的时候，使用的Transformer中encoder层的输出值，一般选择倒数第二层的输出值，这一层向量表示效果最好。也就是说Transfomer层的encoder层输出的**H**是考虑了上下文单词后得到当前单词的向量表示，是上下文相关的。而我们还有个得到输入的向量的表示的部分，通过input_ids得到输入的向量表示**E**，也就是Embeddding层的处理。BERT中的E和H的维度是相等的，E的维度会随H的维度的变大而变大，例如BERT-Large模型中H为1024，E也为1024，这是完全没有必要的，因为我们最终要得到的是H，只要保证H的维度是要求的维度的就可以了。E的维度是可变的，而E是和词表大小息息相关的，即Embeddding层的参数量为V*E，可以将E调整到一个较小的维度，进行优化降低参数量，再通过E*H的变换，将E的维度变换到H的维度。总的参数量也变到了V*E+E*H。（原参数量为V*E）

1. 把词向量维度和注意力hidden*size脱钩（bert里词向量维度=注意力的hidden_size）*

2. 1. *词向量只是表示词汇信息，所以维度过高也没有用*
   2. *注意力的[hidden_size](https://www.zhihu.com/search?q=hidden_size&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A855558666})则要学习到上下文表征信息，所以提高这个参数对模型性能有用*
   3. *实际方法就是词[向量维度](https://www.zhihu.com/search?q=向量维度&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A855558666})=E， 然后再用一个线性变换转换成维度=hidden_size*
   4. *比如10万词汇，E=100，hidden_size=1000,，线性变换矩阵=100×1000 全部参数量=10万×100 + 100×1000 = 1010万*
   5. *上述情况原bert词向量矩阵参数量=10万×1000=1亿*

### Cross-layer parameter sharing

The default decision for ALBERT is to share all parameters across layers

only sharing feed-forward network (FFN) parameters across layers, or only sharing attention parameters.

Cross-layer Parameter Sharing是共享所有层的参数，Transfomer层的encoder部分的参数主要为attention参数和FeedForward的参数，当然LateyNorm也有要学习的参数，不过参数量也别少了。Cross-layer Parameter Sharing主要是共享attention部分的参数和FeedForward部分的参数。这样就大大减少了参数量，但是参数量共享，效果也会下降，

### Inter-sentence coherence loss

In addition to the masked language modeling (MLM) loss, BERT uses an additional loss called next-sentence prediction (NSP)

Sentence Order Prediction是对BERT的NSP预训练进行优化。RoBerta也提出了NSP的预训练效果不是很好，直接将NSP的预训练任务直接去掉了。NSP预训练任务将Topic Prediction和Coherence prediction融合起来了，只要判断两个句子是不是一个Topic的就能对预训练任务出个大概的结果了。Topic Prediction任务非常简单，大大降低了学习的难度。论文通过将负样本换成同一篇文章中的两个[逆序句子](https://www.zhihu.com/search?q=逆序句子&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"268130746"})，来消除Topic prediction，提升预训练任务的学习效果。

*用SOP替换NSP*

1. *NSP：下一句预测， 正样本=上下相邻的2个句子，负样本=随机2个句子*
2. *SOP：句子顺序预测，正样本=正常顺序的2个相邻句子，负样本=调换顺序的2个相邻句子*
3. *NSP任务过于简单，只要模型发现两个句子的主题不一样就行了，所以SOP预测任务能够让模型学习到更多的信息*

## 参数量减少主要靠的是共享参数

这种策略相当于把12个完全相同的层摞起来，只共享attention的参数可以降维也能保持性能不变但是ALBERT把FFN也共享参数了

用SOP补偿了一部分因为embedding和FFN共享损失的性能。

SOP将负样本换成了同一篇文章中的两个逆序句子，进而消除topic prediction

*论文里的消融实验的分数也说明[no-share](https://www.zhihu.com/search?q=no-share&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A855558666})的分数是最高的*



1. 2. 