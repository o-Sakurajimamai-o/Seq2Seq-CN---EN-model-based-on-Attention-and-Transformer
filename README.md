# Seq2Seq-CN---EN-model-based-on-Attention-and-Transformer

Dataset：
1. Attention（seq2seq.py）: [Tatoeba](https://www.manythings.org/anki/), choose CN to EN

2. Transformer(Transformer.py)：in the data profile:  cmn-eng

# English-to-Chinese Machine Translation (Seq2Seq & Transformer)

This repository contains PyTorch implementations of two classic neural machine translation models: a Sequence-to-Sequence (Seq2Seq) model with Attention, and a standard Transformer. 

Both models are built from scratch and trained to translate English into Mandarin Chinese. The implementation details heavily reference the [Dive into Deep Learning (d2l)](https://d2l.ai/) textbook.

## Overview

1. **Seq2Seq with Additive Attention (`seq2seq.py`)**
   * **Encoder/Decoder:** Multi-layer GRU.
   * **Attention:** Additive Attention mechanism to handle variable-length context.
   * **Training Details:** Implements Scheduled Sampling to bridge the gap between training (teacher forcing) and inference. Uses a custom Masked Softmax Loss to ignore padding tokens during backpropagation.

2. **Transformer (`Transformer.py`)**
   * **Architecture:** Standard Encoder-Decoder structure.
   * **Components:** Absolute Positional Encoding, Multi-Head Attention (scaled dot-product), Position-wise Feed-Forward Networks (FFN), and AddNorm (residual connections + layer normalization).

## Project Structure

├── cmn-eng/                        # Dataset directory (Tatoeba CN-EN)
├── Attention.py                    # Attention mechanisms (Additive, Dot-product)
├── data_preprocess.py              # Text cleaning and tokenization
├── sec_machine_translation.py      # Vocab building, data loading, and batching
├── seq2seq.py                      # Seq2Seq model definition, training, and inference
├── Transformer.py                  # Transformer model definition, training, and inference
├── seq2seq.pth                     # Saved weights for Seq2Seq (generated after training)
├── Transformer.pth                 # Saved weights for Transformer (generated after training)
└── README.md

## Dataset

The project uses the **Mandarin Chinese - English** bilingual sentence pairs from the [Tatoeba Project](https://www.manythings.org/anki/) (`cmn-eng`).
* Download the dataset, extract it, and place it in the `cmn-eng/` directory at the root of the project.
* The dataloader automatically handles text preprocessing (converting full-width characters to half-width, inserting spaces before punctuation, etc.).

## Requirements

    pip install torch torchvision
    pip install d2l
    pip install pandas

*Note: A CUDA-enabled GPU is highly recommended for training. The scripts automatically detect and use the GPU via `d2l.try_gpu()`.*

## Quick Start

To train the Seq2Seq model and run some sample predictions:

    python seq2seq.py

To train the Transformer model:

    python Transformer.py

*By default, the scripts will save the model weights (`.pth`) to the local directory after training is complete.*


本项目是一个基于深度学习的英译中（English to Chinese）机器翻译系统。项目中从零开始实现了两种经典的自然语言处理架构：基于 Attention 机制的 Seq2Seq 模型，以及完全基于自注意力机制的 Transformer 模型。

本项目代码大量参考并使用了 [Dive into Deep Learning (d2l)](https://d2l.ai/) 的经典实现方式。

---

## 概览

项目中包含了以下两个核心模型的独立实现与训练：

1. **Seq2Seq with Attention (`seq2seq.py`)**
   * **编码器 (Encoder):** 多层 GRU (Gated Recurrent Unit)。
   * **解码器 (Decoder):** 带有加性注意力机制 (Additive Attention) 的多层 GRU。
   * **特点:** 结合了 Scheduled Sampling（计划采样）进行训练优化，并实现了带有 Mask 的交叉熵损失函数。

2. **Transformer (`Transformer.py`)**
   * **架构:** 标准的 Transformer Encoder-Decoder 结构。
   * **核心组件:** 绝对位置编码 (Positional Encoding)、多头注意力机制 (Multi-Head Attention)、基于缩放点积的注意力、前馈神经网络 (FFN) 以及残差连接与层归一化 (AddNorm)。

---

## 文件结构

```text
├── cmn-eng/                        # 数据集文件夹 (Tatoeba CN-EN)
├── Attention.py                    # 存放不同的注意力机制组件 (Additive, Dot-product)
├── data_preprocess.py              # 数据清理与预处理脚本
├── sec_machine_translation.py      # 词表构建、数据加载、迭代器生成等核心数据模块
├── seq2seq.py                      # Seq2Seq (GRU + Attention) 模型的构建、训练与推理
├── Transformer.py                  # Transformer 模型的构建、训练与推理
├── seq2seq.pth                     # 训练好的 Seq2Seq 模型权重 (需本地生成)
├── Transformer.pth                 # 训练好的 Transformer 模型权重 (需本地生成)
└── README.md                       # 项目说明文档
