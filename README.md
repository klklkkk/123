# 多模态情感分析

## 项目简介

本项目旨在通过融合文本和图像两种模态的信息，实现情感分类任务。具体来说，给定配对的文本和图像，模型将预测对应的情感标签，分类为positive、neutral或negative。项目基于预训练的BERT和ResNet50模型，设计并实现了一个多模态融合模型，以提升情感分类的准确性。

## 环境配置

在开始之前，请确保您的系统已安装以下环境和依赖项。

### 1. Python 版本

- Python 3.7 或以上

### 2. 安装依赖

建议使用 `virtualenv` 或 `conda` 创建虚拟环境。

使用 `pip` 安装所需依赖：

```bash
pip install -r requirements.txt
