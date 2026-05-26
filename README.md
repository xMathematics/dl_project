# 深度学习从零实现 (DL From Scratch)

> 纯 NumPy 实现的深度学习框架，从单层神经网络到 Transformer，**完全手动实现**每一步的前向传播和反向传播。

## 项目目标

- ✅ **不依赖 PyTorch / TensorFlow** — 仅使用 NumPy
- ✅ **模块化设计** — 层、激活函数、损失函数、优化器完全解耦
- ✅ **教学友好** — 代码简洁清晰，注释丰富
- ✅ **循序渐进** — 从最简单的神经元到现代 Transformer

## 项目结构

```
dl_from_scratch/
├── core/           # 基础组件
│   ├── initializers.py   # 权重初始化 (He, Xavier 等)
├── layers/         # 网络层
│   ├── base.py           # 基类 Layer
│   ├── dense.py          # 全连接层
│   ├── convolution.py    # 卷积层 (Conv2D)
│   ├── pooling.py        # 池化层 (MaxPool2D, AvgPool2D)
│   ├── flatten.py        # 展平层
│   ├── dropout.py        # Dropout 层
│   ├── normalization.py  # 批/层归一化
│   ├── embedding.py      # Embedding 层
│   ├── attention.py      # 自注意力 & 多头注意力
│   ├── transformer.py    # Transformer 块 & 位置编码
│   └── sequential.py     # Sequential 容器
├── activations/    # 激活函数
│   └── functions.py      # ReLU, Sigmoid, Tanh, Softmax 等
├── losses/         # 损失函数
│   └── functions.py      # MSE, CrossEntropy, BCE 等
├── optimizers/     # 优化器
│   └── sgd.py            # SGD, Momentum, Adam, RMSprop
├── models/         # 模型封装
│   └── model.py          # 训练/预测/评估接口
└── utils/          # 工具
    ├── data.py           # 数据加载
    └── metrics.py        # 评估指标
```

## 示例

| 示例 | 内容 | 关键词 |
|------|------|--------|
| `01_single_neuron.py` | 单神经元 — 逻辑回归 | 单层神经网络、Sigmoid、二分类 |
| `02_multi_layer_perceptron.py` | 多层感知机 — 分类 | 全连接、反向传播、决策边界 |
| `03_cnn_mnist.py` | CNN — MNIST 手写数字识别 | 卷积、池化、图像分类 |
| `04_transformer_demo.py` | Transformer — 序列任务 | 自注意力、多头注意力、位置编码 |
| `05_attention_visualization.py` | 注意力可视化 | 注意力权重、热力图 |

## 快速开始

```bash
pip install numpy matplotlib scikit-learn
python examples/01_single_neuron.py
```
