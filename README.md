# 深度学习从零实现 (DL From Scratch)

> 纯 NumPy 实现的深度学习 + 机器学习框架，**完全手动实现**每一步的数学推导和算法逻辑。

## 项目目标

- ✅ **不依赖 PyTorch / TensorFlow / sklearn** — 仅使用 NumPy
- ✅ **模块化设计** — 各组件完全解耦
- ✅ **教学友好** — 代码简洁清晰，注释丰富
- ✅ **循序渐进** — 从传统 ML 到现代 Transformer

## 项目结构

```
dl_from_scratch/
├── ml/             # 传统机器学习算法
│   ├── linear_regression.py   # 线性回归 (闭式解 + 梯度下降)
│   ├── logistic_regression.py # 逻辑回归
│   ├── knn.py                 # K 近邻
│   ├── kmeans.py              # K 均值聚类
│   ├── decision_tree.py       # 决策树 (CART)
│   ├── naive_bayes.py         # 高斯朴素贝叶斯
│   ├── pca.py                 # 主成分分析
│   └── svm.py                 # 支持向量机 (SMO)
├── core/           # 基础组件
│   └── initializers.py   # 权重初始化 (He, Xavier 等)
├── layers/         # 深度学习网络层
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
│   └── functions.py      # ReLU, Sigmoid, Tanh, Softmax, GELU 等
├── losses/         # 损失函数
│   └── functions.py      # MSE, CrossEntropy, BCE, L1
├── optimizers/     # 优化器
│   └── sgd.py            # SGD, Momentum, Adam, RMSprop
├── models/         # 模型封装
│   └── model.py          # 训练/预测/评估接口
└── utils/          # 工具
    ├── data.py           # 数据生成
    └── metrics.py        # 评估指标
```

## 示例

### 机器学习

| 示例 | 内容 | 关键词 |
|------|------|--------|
| `06_linear_regression.py` | 线性回归 — 房价预测 | 正规方程、梯度下降、R² |
| `07_knn.py` | K 近邻 — 分类 | 惰性学习、距离度量、K 值 |
| `08_kmeans.py` | K 均值聚类 | 无监督、K-Means++、肘部法则 |
| `09_decision_tree.py` | 决策树 — 分类 | 基尼不纯度、递归分裂、剪枝 |
| `10_pca.py` | PCA 降维 | 特征值分解、方差解释比例 |

### 深度学习

| 示例 | 内容 | 关键词 |
|------|------|--------|
| `01_single_neuron.py` | 单神经元 — AND 逻辑门 | 单层神经网络、Sigmoid、BCE |
| `02_multi_layer_perceptron.py` | 多层感知机 — 分类 | 全连接、反向传播、决策边界 |
| `03_cnn_mnist.py` | CNN — MNIST 手写数字识别 | 卷积、池化、图像分类 |
| `04_transformer_demo.py` | Transformer — 序列建模 | 自注意力、多头注意力、位置编码 |
| `05_attention_visualization.py` | 注意力可视化 | 注意力权重、QKV、因果掩码 |

## 快速开始

```bash
pip install numpy matplotlib
python examples/06_linear_regression.py      # 从传统 ML 开始
python examples/01_single_neuron.py          # 或从深度学习开始
```

> **注意**: `scikit-learn` 仅被 `03_cnn_mnist.py` 用于加载 MNIST 数据集，核心实现不依赖它。
> 如需运行 CNN 示例: `pip install scikit-learn`
