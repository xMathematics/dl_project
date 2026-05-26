"""Embedding 层 — 将离散 Token 映射为稠密向量"""

import numpy as np
from .base import Layer
from ..core.initializers import random_normal


class Embedding(Layer):
    """Embedding 层

    Args:
        vocab_size: 词汇表大小
        embedding_dim: 嵌入维度
        weight_initializer: 权重初始化函数
    """

    def __init__(self, vocab_size, embedding_dim, weight_initializer=random_normal):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight_initializer = weight_initializer
        self._init_params()

    def _init_params(self):
        self.params['W'] = self.weight_initializer((self.vocab_size, self.embedding_dim), std=0.1)
        self.grads['W'] = np.zeros_like(self.params['W'])

    def forward(self, x):
        """前向传播 — 根据索引查找嵌入向量

        Args:
            x: shape=(batch_size, seq_length), dtype=int, 每个元素是 token index

        Returns:
            shape=(batch_size, seq_length, embedding_dim)
        """
        self.cache['indices'] = x.copy()
        return self.params['W'][x]

    def backward(self, grad):
        """反向传播 — 梯度回传到对应索引的嵌入向量

        Args:
            grad: shape=(batch_size, seq_length, embedding_dim)

        Returns:
            对输入的梯度 (无法传播到离散索引，返回 None)
        """
        indices = self.cache['indices']
        batch_size, seq_len = indices.shape

        self.grads['W'].fill(0.0)
        for b in range(batch_size):
            for t in range(seq_len):
                idx = indices[b, t]
                self.grads['W'][idx] += grad[b, t]

        return None

    def __repr__(self):
        return f"Embedding(vocab={self.vocab_size}, dim={self.embedding_dim})"
