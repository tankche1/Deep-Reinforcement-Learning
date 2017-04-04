# 最近看的一些论文
## Mask R-CNN
Our method, called Mask R-CNN, extends Faster R-CNN  by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression (Figure 1).
ROIAlign 

## WGAN的两篇论文 
https://zhuanlan.zhihu.com/p/25071913

## Deep Successor Reinforcement Learning
SR由reward predictor 和 successor map组成。
作者提出DSR可以对末端反馈更敏感。并且可以找出瓶颈状态。
## Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation
用DQN实现V(s,g;theta) 状态s，目标g.
然后不断地更换g。不过目标g似乎要人为设定。比较麻烦。
## Hierarchical Reinforcement Learning using Spatio-Temporal Abstractions and Deep Neural Networks
用PCCA+聚类来指引分层学习。
## STOCHASTIC NEURAL NETWORKS FOR HIERARCHICAL REINFORCEMENT LEARNING
不是很懂
## Value Iteration Networks

## Playing FPS Games with Deep Reinforcement Learning
partially observed,
大概就是分为两个模型。一个导航，一个发现物品或敌人后的action。
然后用内部数据训练了game featrue，找到ammo，weapon，enemy。
如果有enemy就切换到action，否则导航找资源。
action用DQN，导航用DRQN。
使用了reward shape，skipped frames，experience replay（sequential updates） 技巧。

## ACTOR-MIMIC DEEP MULTITASK AND TRANSFER REINFORCEMENT LEARNING

## Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization


