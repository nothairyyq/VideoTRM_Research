## Data Set

# Action Recognition on Something-Something V2

https://paperswithcode.com/sota/action-recognition-in-videos-on-something

记录了人类与日常生活中的一些物体之间的动作数据集. 动作类别被分为174类

The 20BN-SOMETHING-SOMETHING V2 dataset is a large collection of labeled video clips that show humans performing pre-defined basic actions with everyday objects. The dataset was created by a large number of crowd workers. It allows machine learning models to develop fine-grained understanding of basic actions that occur in the physical world. It contains 220,847 videos, with 168,913 in the training set, 24,777 in the validation set and 27,157 in the test set. There are 174 labels.

因为视频既包含空域信息，又包含时域信息，所以**时空信息的融合、特征提取**是该领域的重要方向。

3D卷积与2D卷积的区别，就是多了一个时间维度。多张图像组成一个时间序列