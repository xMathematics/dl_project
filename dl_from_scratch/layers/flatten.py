"""Flatten 层 — 将多维输入展平为 2D

通常位于卷积层和全连接层之间:
  Conv2D 输出: (batch, channels, height, width) — 4D
  Dense 输入:  (batch, channels * height * width) — 2D

Flatten 保持 batch 维度不变，将其余所有维度合并为一个特征向量。
"""

import numpy as np
from .base import Layer


class Flatten(Layer):
    """展平层

    将除 batch 维外的所有维度合并为一维。
    例如: (4, 3, 5, 5) -> (4, 75)

    没有可训练参数 (params 为空)。
    """

    def forward(self, x):
        """前向传播: 展平

        x.reshape(batch_size, -1): 将第 1 维之后的所有维度合并
        - x.shape[0]: batch 大小
        - -1: NumPy 自动计算剩余维度数（原始总元素数 / batch_size）

        参数:
            x: ndarray, 输入, shape=(batch, d1, d2, ..., dn)

        返回:
            out: ndarray, 输出, shape=(batch, d1*d2*...*dn)
        """
        # 缓存原始形状，反向传播时恢复
        self.cache['shape'] = x.shape
        batch_size = x.shape[0]
        # reshape(batch_size, -1): 保持 batch 维，其余合并
        return x.reshape(batch_size, -1)

    def backward(self, grad):
        """反向传播: 恢复到原始形状

        grad.reshape(original_shape): 将展平的梯度恢复为原始多维形状
        这样梯度可以正确地回传到卷积/池化层。

        参数:
            grad: ndarray, 展平的梯度, shape=(batch, flattened_features)

        返回:
            dx: ndarray, 恢复形状后的梯度, shape=(batch, d1, d2, ..., dn)
        """
        original_shape = self.cache['shape']
        # reshape 恢复为前向传播时的输入形状
        return grad.reshape(original_shape)

    def __repr__(self):
        return "Flatten()"
