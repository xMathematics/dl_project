"""Dropout 层 — 防止过拟合的正则化技术

原理 (Srivastava et al., 2014):
  训练时随机"丢弃"一部分神经元（将其输出置零），
  迫使网络学习冗余表示，防止神经元之间过度依赖。

  重要: 训练时用 keep_prob 缩放保留的神经元输出。
  这样在推理时（所有神经元都激活），输出值大小不变。

  Dropout 可以理解为训练了指数多个子网络的集成模型。
"""

import numpy as np
from .base import Layer


class Dropout(Layer):
    """Dropout 层

    训练时以概率 rate 随机丢弃神经元，评估时全部保留。

    参数:
        rate: float, 丢弃概率 (0~1)
              rate=0.1 表示丢弃 10% 的神经元
              rate=0.5 表示丢弃 50%（常用值）
    """

    def __init__(self, rate=0.1):
        super().__init__()
        # rate: 神经元被丢弃的概率
        self.rate = rate

    def forward(self, x):
        """Dropout 前向传播

        训练模式: 随机置零 + 缩放
        评估模式: 不做任何操作 (恒等映射)

        参数:
            x: ndarray, 输入

        返回:
            out: ndarray, 与 x 形状相同
        """
        if self.training:
            # keep_prob = 1 - rate: 神经元保留的概率
            keep_prob = 1.0 - self.rate

            # np.random.binomial(1, keep_prob, size=x.shape):
            # 伯努利采样，每个位置以概率 keep_prob 生成 1，否则 0
            # 结果 mask 是一个 0/1 矩阵
            mask = np.random.binomial(1, keep_prob, size=x.shape)

            # 用 keep_prob 缩放: 保持训练和推理时的期望输出一致
            # E[masked_x] = keep_prob * x / keep_prob = x (推理时)
            mask = mask / keep_prob

            # 保存 mask 供反向传播使用
            self.cache['mask'] = mask
            # 应用 mask: 保留的位置乘以 1/keep_prob，丢弃的位置为 0
            return x * mask

        # 评估模式: 直接通过，不做任何处理
        return x

    def backward(self, grad):
        """Dropout 反向传播

        训练模式: 梯度只通过 mask 中为 1 的位置传播
        评估模式: 梯度直接通过

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        if self.training:
            # 从 cache 读取前向传播时的 mask
            mask = self.cache['mask']
            # 只有 mask=1 (保留) 的位置梯度才能通过
            return grad * mask
        # 评估模式: 梯度直通
        return grad

    def __repr__(self):
        return f"Dropout(rate={self.rate})"
