"""权重初始化器"""

import numpy as np


def he_normal(shape, seed=None):
    """He 正态分布初始化 (适用于 ReLU)"""
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0] if len(shape) > 1 else 1
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def he_uniform(shape, seed=None):
    """He 均匀分布初始化 (适用于 ReLU)"""
    if seed is not None:
        np.random.seed(seed)
    fan_in = shape[0] if len(shape) > 1 else 1
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape, seed=None):
    """Xavier/Glorot 正态分布初始化 (适用于 Tanh/Sigmoid)"""
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape[0], shape[-1] if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def xavier_uniform(shape, seed=None):
    """Xavier/Glorot 均匀分布初始化"""
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape[0], shape[-1] if len(shape) > 1 else shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


def random_normal(shape, mean=0.0, std=0.01, seed=None):
    """正态分布随机初始化"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(*shape) * std + mean


def random_uniform(shape, limit=0.05, seed=None):
    """均匀分布随机初始化"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-limit, limit, size=shape)


def zeros(shape):
    """零初始化"""
    return np.zeros(shape)


def ones(shape):
    """一初始化"""
    return np.ones(shape)
