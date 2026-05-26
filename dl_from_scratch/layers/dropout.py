"""Dropout 层 — 防止过拟合"""

import numpy as np
from .base import Layer


class Dropout(Layer):
    """Dropout 层: 训练时随机丢弃部分神经元

    Args:
        rate: 丢弃概率 (p=0.1 表示丢弃 10%)
    """

    def __init__(self, rate=0.1):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if self.training:
            keep_prob = 1.0 - self.rate
            mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
            self.cache['mask'] = mask
            return x * mask
        return x

    def backward(self, grad):
        if self.training:
            mask = self.cache['mask']
            return grad * mask
        return grad

    def __repr__(self):
        return f"Dropout(rate={self.rate})"
