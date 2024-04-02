# 图神经网络

## 图神经网络核心步骤:
  * step1 构造图(节点，特征，编号)
  * step2 采样邻居节点（均匀采样/重要性采样）聚合邻居节点，获取每个节点的embedding
  * step3 定义损失函数，训练模型（一般就是2, 3层, 非线性变化 & dropout & l2_norm）
  * step4 获取每个节点的embedding或者对应的label


## 相关论文
  * GraphSAGE论文 Inductive Representation Learning on Large Graphs
  * lightGCN论文 LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
  * pinSAGE论文 Graph Convolutional Neural Networks for Web-Scale Recommender Systems
  * PDN论文 Path-based Deep Network for Candidate Item Matching in Recommenders
  * lightGCN.py中几种损失函数的优化方法
    * SSM损失函数 On the Effectiveness of Sampled Softmax Loss for Item Recommendation
    * 正负样本融合 MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems

