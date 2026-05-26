"""优化器 — SGD, Momentum, Adam, RMSprop"""

import numpy as np


class Optimizer:
    """优化器基类"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """更新参数

        Args:
            params: 参数字典 {name: ndarray}
            grads: 梯度字典 {name: ndarray}
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


class SGD(Optimizer):
    """随机梯度下降

    W = W - lr * dW
    """

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def step(self, params, grads):
        for k in params:
            grad = grads[k] + self.weight_decay * params[k]
            params[k] -= self.lr * grad


class SGDWithMomentum(Optimizer):
    """带动量的 SGD

    v = momentum * v - lr * dW
    W = W + v
    """

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.velocities:
                self.velocities[k] = np.zeros_like(params[k])

            grad = grads[k] + self.weight_decay * params[k]
            self.velocities[k] = self.momentum * self.velocities[k] - self.lr * grad
            params[k] += self.velocities[k]


class Adam(Optimizer):
    """Adam 优化器

    m = beta1 * m + (1 - beta1) * dW
    v = beta2 * v + (1 - beta2) * dW^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步

    def step(self, params, grads):
        self.t += 1

        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            grad = grads[k] + self.weight_decay * params[k]

            # 更新偏置矩估计
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad ** 2)

            # 偏差校正
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # 更新参数
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop 优化器

    v = decay * v + (1 - decay) * dW^2
    W = W - lr * dW / (sqrt(v) + epsilon)
    """

    def __init__(self, lr=0.001, decay=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr)
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}

    def step(self, params, grads):
        for k in params:
            if k not in self.cache:
                self.cache[k] = np.zeros_like(params[k])

            grad = grads[k] + self.weight_decay * params[k]
            self.cache[k] = self.decay * self.cache[k] + (1 - self.decay) * (grad ** 2)
            params[k] -= self.lr * grad / (np.sqrt(self.cache[k]) + self.epsilon)
