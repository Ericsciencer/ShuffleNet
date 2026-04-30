# ShuffleNet
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result

<img width="1015" height="440" alt="image" src="https://github.com/user-attachments/assets/679792f9-2109-40ed-98d8-f697f41307e4" />


---

## 简介
ShuffleNet 是由旷视科技团队于 2017 年提出的**轻量化深度卷积神经网络**，相关成果发表于《ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices》，专为移动端、嵌入式等算力受限设备设计。传统轻量化网络大量使用分组卷积后会出现**组间信息隔离、特征流通性差**的问题，ShuffleNet 创新性提出**通道混洗（Channel Shuffle）** 核心机制，在大幅降低参数量与计算量的同时，保证网络特征表达能力。它凭借分组卷积+通道混洗+瓶颈单元的极简设计，在 ImageNet 分类任务上以极低计算开销达到媲美常规 CNN 的精度，突破了传统卷积网络在移动端部署的算力瓶颈。其分组卷积通道打乱、轻量化瓶颈残差单元的设计思想，成为后续 MobileNet、GhostNet 等一众轻量网络的标配范式，广泛应用于图像分类、目标检测、语义分割等移动端视觉任务。

## 架构
ShuffleNet 核心架构为**多阶段堆叠式轻量化卷积神经网络**，整体分为「初始卷积下采样模块」「多阶段 ShuffleNet 特征提取模块」和「全局池化+全连接分类模块」三大部分，原论文标准输入为 224×224 分辨率的 3 通道 RGB 图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
- **初始卷积下采样模块**：采用 3×3 标准卷积核、步长 2 进行初步特征提取与尺寸压缩，搭配 BN 归一化与 ReLU 激活，随后使用 3×3 最大池化进一步降维，快速提取底层边缘、纹理基础特征，同时缩减特征图尺寸，减少后续计算开销。
- **多阶段特征提取模块（Stage2/Stage3/Stage4）**：网络主体由 3 个 Stage 堆叠构成，每个 Stage 由若干 **ShuffleNet 基本单元** 组成；每个单元采用**瓶颈结构**：1×1 分组卷积降通道、3×3 深度卷积提取空间特征、1×1 分组卷积升通道，中间嵌入**通道混洗操作**打破分组卷积信息孤岛；每个 Stage 首个单元步长为 2 负责降采样并扩充通道，其余单元步长为 1 保持特征图分辨率不变，逐层提取中层组合特征与高层语义特征。论文标准配置下 Stage2 含 3 个单元、Stage3 含 7 个单元、Stage4 含 3 个单元，逐级提升通道维度、压缩特征图尺寸。
- **分类输出模块**：后端采用全局自适应平均池化将最后一层特征图压缩为单像素特征向量，直接映射至全连接层，维度匹配分类任务类别数（原论文 ImageNet 任务为 1000 维），输出各类别预测得分。网络全程大量使用分组卷积、深度可分离卷积，舍弃冗余通道与复杂卷积结构，实现极致轻量化。

该架构首次提出**通道混洗**经典设计，完美解决分组卷积信息不流通痛点，以分组卷积+瓶颈残差+通道混洗的组合，在**降低计算量、减少参数量**和**保证特征表达精度**之间实现完美平衡，奠定了移动端轻量化卷积网络的核心设计思路。

<img width="1044" height="454" alt="image" src="https://github.com/user-attachments/assets/864ee39a-291f-485c-aae8-15a224769a82" />
<img width="1020" height="613" alt="image" src="https://github.com/user-attachments/assets/bb5b20f5-91fc-4511-b160-f8c841e3f457" />
<img width="1020" height="494" alt="image" src="https://github.com/user-attachments/assets/f48a4f18-2223-469f-b92e-eab22e14b25b" />


**注意**：我们使用的是数据集 CIFAR-10，它是 10 类彩色数据，并且不同于原文献；由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（调整初始卷积步长、缩减通道缩放系数），但核心架构**通道混洗 + 分组卷积 + 瓶颈Shuffle单元 + 多Stage堆叠**完全保留。

## 数据集
我们使用的是数据集 CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为 32 × 32 ，每个类别有 6000 个图像，数据集中一共有 50000 张训练图片和 10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为 PNG 等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集，如果你想了解该数据集请观看链接：https://cloud.tencent.com/developer/article/2150614

---

## Introduction
ShuffleNet is a lightweight deep convolutional neural network proposed by the Megvii research team in 2017, published in the paper *ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices*. It is specially designed for mobile terminals and embedded devices with limited computing power. Traditional lightweight networks suffer from isolated group information and poor feature flow after extensive use of group convolution. ShuffleNet innovatively proposes the core mechanism of **Channel Shuffle**, which greatly reduces parameters and computational cost while maintaining feature representation capability.

With the minimalist design of group convolution, channel shuffle and bottleneck unit, ShuffleNet achieves accuracy comparable to conventional CNNs on ImageNet classification with extremely low computational overhead, breaking the computing power bottleneck of traditional convolutional networks deployed on mobile devices. Its design ideas of channel permutation for group convolution and lightweight bottleneck residual unit have become standard paradigms for subsequent lightweight networks such as MobileNet and GhostNet, widely applied in mobile visual tasks including image classification, object detection and semantic segmentation.

## Architecture
ShuffleNet adopts a multi-stage stacked lightweight convolutional neural network architecture, divided into three main parts: **initial convolution downsampling module**, **multi-stage ShuffleNet feature extraction module**, and **global pooling + fully connected classification module**. The original paper takes 224×224 RGB images as standard input and outputs predicted probabilities for classification categories. The detailed structure and design are as follows:

- **Initial Convolution Downsampling Module**: A 3×3 standard convolution with stride=2 is used for preliminary feature extraction and size compression, followed by BN normalization and ReLU activation. A 3×3 max pooling layer further reduces dimension, quickly extracting low-level features such as edges and textures while shrinking feature map size to reduce subsequent computation.

- **Multi-stage Feature Extraction Module (Stage2/Stage3/Stage4)**: The main body consists of three stacked stages, each composed of several **ShuffleNet basic units**. Each unit adopts a bottleneck structure: 1×1 group convolution for channel compression, 3×3 depthwise convolution for spatial feature extraction, and 1×1 group convolution for channel recovery. Channel shuffle is embedded in the middle to eliminate information isolation of group convolution. The first unit of each stage uses stride=2 for downsampling and channel expansion, while remaining units use stride=1 to keep feature resolution unchanged, extracting mid-level combined features and high-level semantic features layer by layer. Under the standard paper configuration, Stage2 contains 3 units, Stage3 contains 7 units, and Stage4 contains 3 units, gradually increasing channel dimensions and compressing feature map size.

- **Classification Output Module**: Global adaptive average pooling compresses the final feature map into a single-pixel feature vector, then maps to the fully connected layer. The output dimension matches the number of classification categories (1000 dimensions for ImageNet in the original paper) and outputs prediction scores for each category. The network extensively adopts group convolution and depthwise separable convolution, abandoning redundant channels and complex convolution structures to achieve extreme lightweight design.

The architecture pioneered the classic **Channel Shuffle** design, perfectly solving the problem of poor information flow in group convolution. Combining group convolution, bottleneck residual structure and channel shuffle, it achieves an optimal balance between reducing computation/parameters and maintaining feature representation accuracy, laying the core design foundation for mobile lightweight convolutional networks.

<img width="1225" height="540" alt="image" src="https://github.com/user-attachments/assets/bd5f6425-4330-448c-ac2d-418c866feeb6" />
<img width="1205" height="554" alt="image" src="https://github.com/user-attachments/assets/579418fc-0000-49ba-99d7-79cf2726d5ce" />
<img width="1234" height="530" alt="image" src="https://github.com/user-attachments/assets/757d129e-2ef6-41ee-ac6d-0f758a4bcdb2" />


**Note:** We use the CIFAR-10 dataset, which is a 10-class color dataset. Unlike the original paper, the image size of CIFAR-10 (32×32) is much smaller than the 224×224 in the original paper. We make minor adaptations to the network structure (adjusting the stride of initial convolution and reducing the channel scale factor), but the core architecture **Channel Shuffle + Group Convolution + Bottleneck Shuffle Unit + Multi-Stage Stacking** is completely retained.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

It differs from common image storage formats, using binary-optimized storage. While we could recreate it as PNG or other image formats, that would result in a very large file size. Our focus is on neural networks, so we won't delve into a detailed analysis of the dataset here.

---
## 原文章 | Original article
Zhang, Xiangyu, Xinyu Zhou, Mengxiao Lin, and Jian Sun. "Shufflenet: An extremely efficient convolutional neural network for mobile devices." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
