"""归一化层 — Batch Normalization & Layer Normalization"""

import numpy as np
from .base import Layer
from ..core.initializers import ones, zeros


class BatchNormalization(Layer):
    """批归一化 (Batch Normalization)

    对每个特征维度，在 batch 维度上进行归一化。

    Args:
        axis: 归一化维度轴 (默认 1，即 channels 维度)
        momentum: 移动平均动量
        epsilon: 数值稳定常数
    """

    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self._initialized = False

    def _init_params(self, num_features):
        self.params['gamma'] = ones((1, num_features, 1, 1))
        self.params['beta'] = zeros((1, num_features, 1, 1))
        # 运行时均值和方差
        self.running_mean = zeros((1, num_features, 1, 1))
        self.running_var = ones((1, num_features, 1, 1))
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        self._initialized = True

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, channels, height, width) 或 (batch_size, features)

        Returns:
            归一化后的输出
        """
        # 确保输入是 4D
        if x.ndim == 2:
            x = x[:, :, np.newaxis, np.newaxis]

        if not self._initialized:
            self._init_params(x.shape[1])

        if self.training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # 更新运行时统计
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            x_centered = x - batch_mean
            x_normalized = x_centered / np.sqrt(batch_var + self.epsilon)

            self.cache['x_centered'] = x_centered
            self.cache['x_normalized'] = x_normalized
            self.cache['batch_var'] = batch_var
            self.cache['x_shape'] = x.shape
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        out = self.params['gamma'] * x_normalized + self.params['beta']
        return out

    def backward(self, grad):
        """反向传播"""
        x_shape = self.cache['x_shape']
        x_centered = self.cache['x_centered']
        x_normalized = self.cache['x_normalized']
        batch_var = self.cache['batch_var']
        batch_size = x_shape[0]
        spatial_size = x_shape[2] * x_shape[3]

        # dL/dgamma = sum(grad * x_normalized, axis=(0,2,3), keepdims=True)
        self.grads['gamma'] = np.sum(grad * x_normalized, axis=(0, 2, 3), keepdims=True)
        self.grads['beta'] = np.sum(grad, axis=(0, 2, 3), keepdims=True)

        # 反向传播通过归一化
        N = batch_size * spatial_size
        var = batch_var + self.epsilon
        sqrt_var = np.sqrt(var)
        inv_sqrt_var = 1.0 / sqrt_var

        g_normalized = grad * self.params['gamma']

        # dx_normalized -> dx
        dx = (1.0 / N) * inv_sqrt_var * (
            N * g_normalized
            - np.sum(g_normalized, axis=(0, 2, 3), keepdims=True)
            - x_normalized * np.sum(g_normalized * x_normalized, axis=(0, 2, 3), keepdims=True)
        )

        return dx

    def __repr__(self):
        return f"BatchNormalization(axis={self.axis})"


class LayerNormalization(Layer):
    """层归一化 (Layer Normalization)

    对每个样本的最后一维进行归一化。
    支持任意维度输入: (batch, features), (batch, seq, d_model) 等。

    Args:
        epsilon: 数值稳定常数
    """

    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self._initialized = False

    def _init_params(self, x):
        """延迟初始化: gamma/beta 形状匹配最后一维"""
        d_model = x.shape[-1]
        # 构建与 x 相同 ndim 的 gamma/beta，除最后一维外均为 1
        shape = tuple(1 for _ in range(x.ndim - 1)) + (d_model,)
        self.params['gamma'] = ones(shape)
        self.params['beta'] = zeros(shape)
        self.grads['gamma'] = np.zeros_like(self.params['gamma'])
        self.grads['beta'] = np.zeros_like(self.params['beta'])
        self._initialized = True

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, ..., d_model)

        Returns:
            归一化后的输出，形状与输入相同
        """
        if not self._initialized:
            self._init_params(x)

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)

        self.cache['x_normalized'] = x_normalized
        self.cache['var'] = var
        self.cache['shape'] = x.shape

        return self.params['gamma'] * x_normalized + self.params['beta']

    def backward(self, grad):
        """反向传播"""
        x_normalized = self.cache['x_normalized']
        var = self.cache['var']
        N = x_normalized.shape[-1]

        # dL/dgamma, dL/dbeta — 在非最后一维上求和
        reduce_axes = tuple(range(grad.ndim - 1))
        self.grads['gamma'] = np.sum(grad * x_normalized, axis=reduce_axes, keepdims=True)
        self.grads['beta'] = np.sum(grad, axis=reduce_axes, keepdims=True)

        g = grad * self.params['gamma']
        sqrt_var = np.sqrt(var + self.epsilon)
        inv_sqrt_var = 1.0 / sqrt_var

        dx = (1.0 / N) * inv_sqrt_var * (
            N * g
            - np.sum(g, axis=-1, keepdims=True)
            - x_normalized * np.sum(g * x_normalized, axis=-1, keepdims=True)
        )

        return dx

    def __repr__(self):
        return "LayerNormalization()"
