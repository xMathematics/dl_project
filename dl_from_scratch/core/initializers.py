"""权重初始化器

权重初始化对深度学习训练至关重要:
- 初始化太小 → 信号在传播中消失 (vanishing)
- 初始化太大 → 信号爆炸 (exploding)
- 合适的初始化加速收敛，提高最终性能

常用策略:
  He 初始化  (适用于 ReLU):    std = √(2/fan_in)
  Xavier 初始化 (适用于 Tanh): std = √(2/(fan_in + fan_out))

  fan_in  = 输入神经元数
  fan_out = 输出神经元数
"""

import numpy as np


def he_normal(shape, seed=None):
    """He 正态分布初始化 (Kaiming He et al., 2015)

    适用于 ReLU 及其变体 (LeakyReLU, PReLU)。
    权重从 N(0, std²) 采样，其中 std = √(2 / fan_in)。

    原理: ReLU 将一半神经元置零，因此方差只需要 Xavier 的一半。

    参数:
        shape: tuple, 权重矩阵的形状，如 (input_dim, output_dim)
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        # np.random.seed(seed): 固定随机种子，确保结果可复现
        np.random.seed(seed)

    # fan_in: 输入神经元数 (shape[0])
    # 对于 Conv2D 的 shape=(out_c, in_c, k_h, k_w)，fan_in = in_c * k_h * k_w
    fan_in = shape[0] if len(shape) > 1 else 1

    # He 初始化标准差: std = √(2 / fan_in)
    # np.sqrt(2.0 / fan_in): 计算标准差
    std = np.sqrt(2.0 / fan_in)

    # np.random.randn(*shape): 生成标准正态分布 N(0,1) 的随机数
    # * shape: 解包元组，如 (3,4) -> randn(3,4)
    # * std: 缩放为标准差 std
    return np.random.randn(*shape) * std


def he_uniform(shape, seed=None):
    """He 均匀分布初始化

    从 U(-limit, limit) 采样，其中 limit = √(6 / fan_in)。
    √6 是 √2 的 √3 倍，因为均匀分布的方差 = limit²/3。

    参数:
        shape: tuple, 权重形状
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    fan_in = shape[0] if len(shape) > 1 else 1

    # 均匀分布的边界: limit = √(6 / fan_in)
    # np.sqrt(6.0 / fan_in): √(6/fan_in)
    limit = np.sqrt(6.0 / fan_in)

    # np.random.uniform(-limit, limit, size=shape): 从 [-limit, limit] 均匀采样
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape, seed=None):
    """Xavier/Glorot 正态分布初始化 (Glorot & Bengio, 2010)

    适用于 Tanh, Sigmoid 等对称激活函数。
    权重从 N(0, std²) 采样，其中 std = √(2 / (fan_in + fan_out))。

    原理: 保持前向和反向传播的方差一致。

    参数:
        shape: tuple, 权重形状
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # fan_in: 输入神经元数
    fan_in = shape[0] if len(shape) > 1 else 1
    # fan_out: 输出神经元数 (shape[-1] 是最后一个维度)
    fan_out = shape[-1] if len(shape) > 1 else shape[0]

    # Xavier 标准差: std = √(2 / (fan_in + fan_out))
    # 使用 fan_in + fan_out 的调和平均
    std = np.sqrt(2.0 / (fan_in + fan_out))

    return np.random.randn(*shape) * std


def xavier_uniform(shape, seed=None):
    """Xavier/Glorot 均匀分布初始化

    从 U(-limit, limit) 采样，其中 limit = √(6 / (fan_in + fan_out))。

    参数:
        shape: tuple, 权重形状
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    fan_in = shape[0] if len(shape) > 1 else 1
    fan_out = shape[-1] if len(shape) > 1 else shape[0]

    # 均匀分布边界: limit = √(6 / (fan_in + fan_out))
    limit = np.sqrt(6.0 / (fan_in + fan_out))

    return np.random.uniform(-limit, limit, size=shape)


def random_normal(shape, mean=0.0, std=0.01, seed=None):
    """正态分布随机初始化 (简单版本)

    权重从 N(mean, std²) 采样，适用于没有特殊要求的场景。

    参数:
        shape: tuple, 权重形状
        mean: float, 正态分布的均值
        std: float, 正态分布的标准差
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # np.random.randn(*shape): 标准正态分布 N(0,1)
    # * std + mean: 缩放平移为 N(mean, std²)
    return np.random.randn(*shape) * std + mean


def random_uniform(shape, limit=0.05, seed=None):
    """均匀分布随机初始化

    从 U(-limit, limit) 采样。

    参数:
        shape: tuple, 权重形状
        limit: float, 边界值
        seed: int 或 None, 随机种子

    返回:
        ndarray: 初始化后的权重矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # np.random.uniform: 从均匀分布 U(-limit, limit) 采样
    return np.random.uniform(-limit, limit, size=shape)


def zeros(shape):
    """零初始化 — 将所有权重初始化为 0

    通常用于偏置 (bias)，不推荐用于权重 (会导致对称性)。

    参数:
        shape: tuple, 输出形状

    返回:
        ndarray: 全零数组
    """
    # np.zeros(shape): 创建指定形状的全零数组
    return np.zeros(shape)


def ones(shape):
    """一初始化"""
    return np.ones(shape)
