# 感知机与神经网络 - Manim 动画项目  

本项目基于鸢尾花分类与感知机模型的经典示例，逐步扩展到神经网络的原理和应用。通过 Manim 制作动画，帮助读者可视化并理解机器学习中从线性到非线性、从单层感知机到深层网络的过程。  

---  

## 目录  

1. [项目简介](#项目简介)  
2. [文件结构](#文件结构)  
3. [使用环境](#使用环境)  
4. [使用方法](#使用方法)  
5. [项目章节说明](#项目章节说明)  
6. [贡献与反馈](#贡献与反馈)  

---  

## 项目简介  

- **主要思路**  
  - 以鸢尾花(Iris)数据集作为分类任务的示例，介绍感知机(Perceptron)模型的原理、构建与训练流程。  
  - 在感知机无法解决异或(XOR)问题的基础上，引出多层感知机(MLP)及神经网络的概念与应用场景。  
- **目标受众**  
  - 想要快速了解感知机与神经网络基础原理的学习者。  
  - 需要可视化动画提升理解的机器学习初学者或教学工作者。  

通过简单到复杂的逐步演示，本项目有助于加深对线性可分、非线性映射、多层网络结构等概念的理解。  

---  

## 文件结构  

为更好地管理思维导图中的各个知识节点，每个章节及重要子节点都分配了一个独立的 `.py` 文件。建议在同一文件夹中存放这些文件，并遵循以下命名约定：  

```plaintext  
.  
├── 01_Introduction.py                   # 一、引言  
├── 02_IrisData.py                       # 二、鸢尾花数据  
├── 03_Perceptron_Overview.py            # 三、感知机概述  
├── 03_1_Perceptron_ClassPrediction.py   # 3.1.1 判断样本类别  
├── 03_2_Perceptron_ClassificationLogic.py  
├── 03_3_Perceptron_SeparationLine.py  
├── 03_4_Perceptron_FindWeights.py  
├── 03_5_Perceptron_OptimizeLoss.py  
├── 03_6_Perceptron_ParamAdjustment.py  
├── 03_7_Perceptron_HighDimension.py  
├── 03_8_Perceptron_ModelBuilding.py  
├── 03_9_Perceptron_XORProblem.py        # 3.4 异或问题  
├── 04_MultilayerPerceptron.py           # 四、多层感知机(MLP)  
├── 05_NeuralNetworks.py                 # 五、神经网络  
└── README.md