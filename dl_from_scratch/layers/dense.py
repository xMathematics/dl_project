"""全连接层 (Dense / Fully Connected / Linear)

数学原理:
  y = x @ W + b

  其中:
  - x: 输入, shape=(..., input_dim)
  - W: 权重矩阵, shape=(input_dim, units)
  - b: 偏置向量, shape=(1, units)
  - y: 输出, shape=(..., units)

  @ 是矩阵乘法, 将输入从 input_dim 维映射到 units 维。
"""

import numpy as np
from .base import Layer
from ..core.initializers import he_normal, zeros


class Dense(Layer):
    """全连接层 (Dense / Fully Connected)

    执行线性变换 y = x @ W + b，将输入特征映射到输出空间。
    是神经网络中最基本的层。

    参数:
        units: int, 输出神经元数量 (决定输出的维度)
        use_bias: bool, 是否使用偏置项 b (默认 True)
        weight_initializer: 权重初始化函数 (默认 He 初始化)
        bias_initializer: 偏置初始化函数 (默认零初始化)
    """

    def __init__(self, units, use_bias=True,
                 weight_initializer=he_normal, bias_initializer=zeros):
        # 调用父类 Layer 的 __init__，初始化 params, grads, cache
        super().__init__()
        # units: 输出神经元的数量，即 W 的第二维度
        self.units = units
        # use_bias: 是否包含偏置项 b
        self.use_bias = use_bias
        # weight_initializer: 权重 W 的初始化策略
        self.weight_initializer = weight_initializer
        # bias_initializer: 偏置 b 的初始化策略
        self.bias_initializer = bias_initializer
        # _initialized: 延迟初始化标志
        # 在第一次 forward 时才知道 input_dim，所以延迟创建参数
        self._initialized = False

    def _init_params(self, input_dim):
        """延迟初始化可训练参数

        在第一次调用 forward 时，根据输入维度创建权重和偏置。

        weight_shape = (input_dim, units): 将 input_dim 维映射到 units 维
        W @ x: (units, input_dim) @ (input_dim,) -> (units,)
        但我们用 x @ W: (batch, input_dim) @ (input_dim, units) -> (batch, units)

        参数:
            input_dim: int, 输入特征的维度
        """
        weight_shape = (input_dim, self.units)

        # 使用指定的初始化函数创建权重 W
        # he_normal(weight_shape): He 正态分布初始化
        self.params['W'] = self.weight_initializer(weight_shape)

        if self.use_bias:
            # 偏置 b 的形状: (1, units)，广播到每个样本
            # zeros((1, units)): 零初始化偏置
            self.params['b'] = self.bias_initializer((1, self.units))

        # 创建与参数同形的梯度数组，初始全零
        # np.zeros_like: 创建形状相同的全零数组
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])

        self._initialized = True

    def forward(self, x):
        """全连接层前向传播

        y = x @ W + b

        支持任意维度输入，最后一维被当作特征维度:
        - (batch, input_dim) -> (batch, units)
        - (batch, seq_len, input_dim) -> (batch, seq_len, units)

        这是通过将前 n-1 维合并为 batch 维度实现的。

        参数:
            x: ndarray, 输入数据, 最后一维是 input_dim

        返回:
            out: ndarray, 输出数据, shape=(*x.shape[:-1], units)
        """
        # x.shape[-1]: 最后一维的尺寸，即特征维度 input_dim
        input_dim = x.shape[-1]

        # 延迟初始化: 第一次调用时创建参数
        if not self._initialized:
            self._init_params(input_dim)

        # 保存原始形状，供反向传播恢复使用
        original_shape = x.shape
        self.cache['original_shape'] = original_shape

        # 展平除最后一维外的所有维度为 batch 维度
        # reshape(-1, input_dim): -1 表示自动计算该维度
        # 例如 (2, 8, 32) -> (16, 32)
        x_flat = x.reshape(-1, input_dim)
        self.cache['x_flat'] = x_flat

        # 线性变换: y = x @ W
        # @ 是 NumPy 的矩阵乘法运算符 (等价于 np.matmul)
        # (N, input_dim) @ (input_dim, units) -> (N, units)
        out = x_flat @ self.params['W']

        # 加偏置: y = x @ W + b
        if self.use_bias:
            # b 的形状为 (1, units)，通过广播加到每个样本
            # 广播: (N, units) + (1, units) -> (N, units)
            out += self.params['b']

        # 恢复输出形状: 将展平的 batch 维还原
        # out_shape = (原 batch, 原 seq_len, ..., units)
        out_shape = original_shape[:-1] + (self.units,)
        return out.reshape(out_shape)

    def backward(self, grad):
        """全连接层反向传播

        计算三个梯度:
        1. dL/dW = x^T @ dL/dy  (权重梯度)
        2. dL/db = sum(dL/dy)    (偏置梯度)
        3. dL/dx = dL/dy @ W^T   (输入梯度，传给上一层)

        参数:
            grad: ndarray, 上游梯度 dL/dy, shape=(..., units)

        返回:
            dx: ndarray, 对输入的梯度 dL/dx, shape=(..., input_dim)
        """
        original_shape = self.cache['original_shape']
        x_flat = self.cache['x_flat']

        # 将梯度展平: 与 forward 中的 x_flat 对应
        # grad.reshape(-1, self.units): 合并除 units 外的所有维度
        grad_flat = grad.reshape(-1, self.units)

        # ---- 计算参数梯度 ----

        # dL/dW = x^T @ dL/dy
        # x_flat.T: (input_dim, N)
        # grad_flat: (N, units)
        # 结果: (input_dim, units) 与 W 形状相同
        self.grads['W'] = x_flat.T @ grad_flat

        if self.use_bias:
            # dL/db = Σ dL/dy (对所有样本求和)
            # np.sum(grad_flat, axis=0): 在 batch 维度求和
            # keepdims=True: 保持 (1, units) 形状
            self.grads['b'] = np.sum(grad_flat, axis=0, keepdims=True)

        # ---- 计算对输入的梯度 ----

        # dL/dx = dL/dy @ W^T
        # grad_flat: (N, units)
        # W.T: (units, input_dim)
        # 结果: (N, input_dim)
        dx = grad_flat @ self.params['W'].T

        # 恢复输入的原始形状
        return dx.reshape(original_shape)

    def __repr__(self):
        """返回层的字符串表示"""
        return f"Dense(units={self.units}, bias={self.use_bias})"
