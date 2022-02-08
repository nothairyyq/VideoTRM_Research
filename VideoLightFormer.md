# VideoLightFormer

## Abstract

 extend the 2D convolutional Temporal Segment Network with transformers, while maintaining spatial and temporal video structure throughout the entire model.

Our method differs from them by keeping the transformer models small, but leveraging full spatiotemporal feature structure

## Introducation

we apply a factorized transformer approach that retains all spatiotemporal feature structure

Video Light Former extends the base Temporal Segment Network with lightweight spatial transformers

Overall, our contributions are:

1. A novel action recognition architecture VLF that is based on 2D convolutions and lightweight transformers to leverage spatiotemporal feature structure in an efficient manner
2. 

## Related Works

- TimeSformer and ViViT linearly initialize a video into a sequence of vectors (Tokenization), and then feed them into a large transformer model.

![image-20220126115530773](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220126115530773.png)

- VTN instead 2D convolutionally transforms each frame into a single embedding, and then feeds this sequence into a small transformer
- VLF 2D convolutionally transforms each frame into spatial feature-maps that are fed through small, spatial transformers. All resulting features are finally fed into a small, spatiotemporal transformer, which has higher modeling capabilities than VTN because it has access to both temporal and spatial input structure

TSM outperforms even the biggest TimeSformer on the temporally-demanding Something-Something dataset [10], suggesting that TimeSformer and ViViT are poor choices when temporal modeling ability is required

 In VLF (Figure 2c), we instead apply a factorized approach, where spatial transformers first improve the CNN feature-maps of each frame, after which a final spatiotemporal transformer takes in all features and produces outputs

![image-20220126135932885](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20220126135932885.png)

## Model

$VLF(X) = MLP(h(g(f(x_{1})),g(f(x_{2})),...,g(f(x_{k}))))$

Four consecutive parts:

> $f(x)$: a 2D convolutional feature extraction backbone
>
> ​	Transforms each fram into a sequence of features $s_{i}$

> $g(x)$: spatial transformer,
>
> ​	use attention to improve the features of each frame

> $h(x)$: spatiotemporal transformer 
>
> ​	model the entire set of video features $z$

### 1. cnn backbone

 VideoLightFormer positions itself in the middle of the two, by retaining spatial structure, but at a low resolution. This only fractionally increases the required computation, but adds a large amount of modeling capability in the following transformers.

### 2. Spatial Transformer

The Detection Transformer (DETR) [3] shows promising results in object detection by propagating an image through a 2D CNN and then a spatial transformer.

Spatial Transformer $g(x)$ perform this step separately on each frame $\widehat{x}_{i}\in \mathbb{R}^{C \times H\times W}$

Unroll the spatial dimensions $H$ and $W$ of $\widehat{x}_{i}$ into a sequence $s_{i} \in \mathbb R ^{C\times HW}$

​					$\widehat{s}_{i} = g(s_{i}) \in \mathbb R^{C\times HW}$

Overall, while extending the DETR approach to videos, g(x) introduces minimal computational overhead, because of the reduced feature resolution at this stage and because of processing each frame separately

### 3. Spatiotemporal Transformer

z = concat(s1,s2...sk)

Given z, the transformer h(x) then has access to both spatial and temporal structure and can model interactions between these two without spatial or temporal ambiguity.

- Difference VTN:

  VTN的transformer只能访问input frame的单个向量，并且依赖于将每个frame的整个spatial信息压缩成单个向量

- Difference TimeSFormer/ ViViT:

  三个model都是在这一阶段使用spatialtemporal transformer

  但是，VLF offloads a large amount of modeling onto the convolutional backbone f(x) and spatial transformer g(x)


