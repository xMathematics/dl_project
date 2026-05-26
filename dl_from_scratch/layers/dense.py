"""全连接层 (Dense / Linear)"""

import numpy as np
from .base import Layer
from ..core.initializers import he_normal, zeros


class Dense(Layer):
    """全连接层: y = x @ W + b

    Args:
        units: 输出神经元数量
        use_bias: 是否使用偏置
        weight_initializer: 权重初始化函数
        bias_initializer: 偏置初始化函数
    """

    def __init__(self, units, use_bias=True,
                 weight_initializer=he_normal, bias_initializer=zeros):
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._initialized = False

    def _init_params(self, input_dim):
        """延迟初始化参数"""
        weight_shape = (input_dim, self.units)
        self.params['W'] = self.weight_initializer(weight_shape)
        if self.use_bias:
            self.params['b'] = self.bias_initializer((1, self.units))
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])
        self._initialized = True

    def forward(self, x):
        """前向传播

        支持任意维度输入，最后一维作为特征维:
          (batch, input_dim) -> (batch, units)
          (batch, seq, input_dim) -> (batch, seq, units)

        Args:
            x: 输入数据，最后一维是 input_dim

        Returns:
            输出数据，形状为 (*x.shape[:-1], units)
        """
        input_dim = x.shape[-1]
        if not self._initialized:
            self._init_params(input_dim)

        # 保存原始形状
        original_shape = x.shape
        self.cache['original_shape'] = original_shape

        # 展平除最后一维外的所有维度
        x_flat = x.reshape(-1, input_dim)
        self.cache['x_flat'] = x_flat

        out = x_flat @ self.params['W']
        if self.use_bias:
            out += self.params['b']

        # 恢复形状
        out_shape = original_shape[:-1] + (self.units,)
        return out.reshape(out_shape)

    def backward(self, grad):
        """反向传播

        Args:
            grad: 上游梯度

        Returns:
            对输入的梯度，形状与 forward 输入相同
        """
        original_shape = self.cache['original_shape']
        x_flat = self.cache['x_flat']

        # 展平梯度
        grad_flat = grad.reshape(-1, self.units)

        # dL/dW = x^T @ grad
        self.grads['W'] = x_flat.T @ grad_flat
        if self.use_bias:
            self.grads['b'] = np.sum(grad_flat, axis=0, keepdims=True)

        # dL/dx = grad @ W^T
        dx = grad_flat @ self.params['W'].T
        return dx.reshape(original_shape)

    def __repr__(self):
        return f"Dense(units={self.units}, bias={self.use_bias})"
