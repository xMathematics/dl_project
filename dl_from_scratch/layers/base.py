"""网络层基类"""

import numpy as np


class Layer:
    """所有网络层的基类

    每个层需要实现:
    - forward(x): 前向传播
    - backward(grad): 反向传播
    """

    def __init__(self):
        self.params = {}      # 可训练参数: {name: ndarray}
        self.grads = {}       # 参数梯度: {name: ndarray}
        self.cache = {}       # 前向传播缓存 (供反向传播使用)
        self.training = True  # 训练/评估模式

    def forward(self, x):
        """前向传播

        Args:
            x: 输入数据, shape=(batch_size, ...)

        Returns:
            输出数据
        """
        raise NotImplementedError

    def backward(self, grad):
        """反向传播

        Args:
            grad: 上游梯度, shape=(batch_size, ...)

        Returns:
            下游梯度 (对输入的梯度)
        """
        raise NotImplementedError

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为评估模式"""
        self.training = False

    def get_params(self):
        """获取所有可训练参数"""
        return self.params

    def get_grads(self):
        """获取所有参数梯度"""
        return self.grads

    def zero_grad(self):
        """清零梯度"""
        for k in self.grads:
            self.grads[k] = np.zeros_like(self.grads[k])

    def num_params(self):
        """计算参数量"""
        total = 0
        for p in self.params.values():
            total += np.prod(p.shape)
        return total

    def __repr__(self):
        return f"{self.__class__.__name__}()"
