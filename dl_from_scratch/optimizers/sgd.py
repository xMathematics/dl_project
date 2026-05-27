"""优化器 — 参数更新算法

优化器根据梯度更新模型参数，是深度学习训练的核心组件。

本模块实现:
- SGD: 最基础的随机梯度下降
- SGDWithMomentum: 带动量的 SGD (加速收敛，抑制震荡)
- Adam: 自适应矩估计 (最常用的优化器)
- RMSprop: 自适应学习率 (对非平稳目标有效)
"""

import numpy as np


class Optimizer:
    """优化器基类

    所有优化器都实现 step(params, grads) 方法:
    1. 接收当前参数和梯度
    2. 更新参数值 (原地修改)
    """

    def __init__(self, lr=0.01):
        # lr (learning rate): 学习率，控制参数更新的步长
        # 太大: 不收敛 (震荡/发散)
        # 太小: 收敛慢 (训练时间长)
        self.lr = lr

    def step(self, params, grads):
        """更新所有参数

        参数:
            params: dict, 模型参数字典 {name: ndarray}
            grads: dict, 梯度字典 {name: ndarray} (与 params 键相同)

        抛出:
            NotImplementedError: 子类必须实现
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr})"


class SGD(Optimizer):
    """随机梯度下降 (Stochastic Gradient Descent)

    最简单的优化器，直接沿梯度反方向更新参数。

    公式:
      W = W - lr * dW

    其中 dW = 梯度 + weight_decay * W (如果使用权重衰减)

    参数:
        lr: float, 学习率
        weight_decay: float, L2 正则化强度 (0 = 不使用)
    """

    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr)
        # weight_decay: L2 正则化 (权重衰减)
        # 相当于在损失函数中添加 λ||W||² 惩罚项
        # 防止权重过大，减少过拟合
        self.weight_decay = weight_decay

    def step(self, params, grads):
        """SGD 参数更新

        对每个参数:
          grad' = dL/dW + λW  (加入权重衰减)
          W = W - lr * grad'

        参数:
            params: dict, 模型参数
            grads: dict, 参数梯度
        """
        for k in params:
            # 加入权重衰减 (L2 正则化)
            # weight_decay * params[k]: L2 惩罚项的梯度
            grad = grads[k] + self.weight_decay * params[k]
            # 参数更新: W -= lr * dW
            params[k] -= self.lr * grad


class SGDWithMomentum(Optimizer):
    """带动量的 SGD (Momentum SGD)

    动量法累积历史梯度，像小球滚下山坡一样加速更新。

    公式:
      v = momentum * v - lr * dW    (速度更新)
      W = W + v                      (参数更新)

    动量 v 是梯度的指数衰减平均:
    - momentum=0.9: 相当于最近 10 步的平均梯度
    - 在平坦区域加速，在震荡区域抑制

    参数:
        lr: float, 学习率
        momentum: float, 动量系数 (通常 0.9 ~ 0.99)
        weight_decay: float, L2 正则化强度
    """

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, params, grads):
        """带动量的 SGD 参数更新

        对每个参数:
          v_new = momentum * v_old - lr * (dL/dW + λW)
          W = W + v_new

        参数:
            params: dict, 模型参数
            grads: dict, 参数梯度
        """
        for k in params:
            # 第一次遇到该参数时初始化速度为零
            if k not in self.velocities:
                # np.zeros_like(params[k]): 创建与参数同形的零数组
                self.velocities[k] = np.zeros_like(params[k])

            # 加入权重衰减的梯度
            grad = grads[k] + self.weight_decay * params[k]

            # 更新速度: v = momentum * v - lr * grad
            # momentum * v: 保留部分历史速度（惯性）
            # - lr * grad: 当前梯度带来的加速度
            self.velocities[k] = self.momentum * self.velocities[k] - self.lr * grad

            # 更新参数: W = W + v
            params[k] += self.velocities[k]


class Adam(Optimizer):
    """Adam 优化器 (Adaptive Moment Estimation)

    Kingma & Ba, 2014. 目前最常用的深度学习优化器。

    结合了:
    - Momentum: 一阶矩 (梯度的指数平均)
    - RMSprop: 二阶矩 (梯度平方的指数平均)
    - 自适应学习率: 每个参数有不同的有效学习率

    公式:
      m = β₁·m + (1-β₁)·dW           # 一阶矩 (梯度均值)
      v = β₂·v + (1-β₂)·(dW)²         # 二阶矩 (梯度方差)
      m̂ = m / (1 - β₁ᵗ)                # 偏差校正
      v̂ = v / (1 - β₂ᵗ)
      W = W - lr · m̂ / (√v̂ + ε)       # 参数更新

    参数:
        lr: float, 学习率 (默认 0.001，通常效果不错)
        beta1: float, 一阶矩衰减率 (默认 0.9)
        beta2: float, 二阶矩衰减率 (默认 0.999)
        epsilon: float, 数值稳定常数 (默认 1e-8)
        weight_decay: float, L2 正则化强度
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr)
        # beta1: 一阶矩的指数衰减率 (控制动量)
        # 越大表示对历史梯度依赖越强
        self.beta1 = beta1
        # beta2: 二阶矩的指数衰减率 (控制自适应学习率)
        # 越大表示对历史梯度平方依赖越强
        self.beta2 = beta2
        # epsilon: 防止除以零的小常数
        self.epsilon = epsilon
        # weight_decay: L2 正则化强度
        self.weight_decay = weight_decay
        # m: 一阶矩 (梯度的指数移动平均) {参数名: ndarray}
        self.m = {}
        # v: 二阶矩 (梯度平方的指数移动平均) {参数名: ndarray}
        self.v = {}
        # t: 时间步计数器，用于偏差校正
        self.t = 0

    def step(self, params, grads):
        """Adam 参数更新

        参数:
            params: dict, 模型参数
            grads: dict, 参数梯度
        """
        # 时间步 +1
        self.t += 1

        for k in params:
            # 首次遇到该参数时初始化 m, v 为零
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            # 加入权重衰减的梯度
            grad = grads[k] + self.weight_decay * params[k]

            # ---- 更新有偏矩估计 ----
            # m = β₁·m + (1-β₁)·grad: 梯度的一阶矩 (均值)
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            # v = β₂·v + (1-β₂)·grad²: 梯度的二阶矩 (未中心化的方差)
            # grad ** 2: 逐元素平方
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad ** 2)

            # ---- 偏差校正 ----
            # 初始时 m, v 偏向 0，通过除以 (1-βᵗ) 校正
            # t 越大，校正因子越接近 1
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # ---- 参数更新 ----
            # W = W - lr * m̂ / (√v̂ + ε)
            # np.sqrt(v_hat): 对二阶矩开方 (标准差)
            # m_hat / (np.sqrt(v_hat) + epsilon): 自适应学习率
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """RMSprop 优化器 (Root Mean Square Propagation)

    Hinton 在 Coursera 课程中提出，对非平稳目标函数效果较好。

    思想: 根据梯度的大小动态调整学习率
    - 梯度大: 降低学习率 (防止震荡)
    - 梯度小: 增大学习率 (加速收敛)

    公式:
      v = β·v + (1-β)·(dW)²     # 梯度平方的移动平均
      W = W - lr · dW / (√v + ε)  # 参数更新

    参数:
        lr: float, 学习率 (默认 0.001)
        decay: float, 衰减率 (默认 0.9)
        epsilon: float, 数值稳定常数
        weight_decay: float, L2 正则化强度
    """

    def __init__(self, lr=0.001, decay=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr)
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = {}

    def step(self, params, grads):
        """RMSprop 参数更新

        对每个参数:
          v = β·v + (1-β)·(dW)²       # 梯度平方的移动平均
          W = W - lr · dW / (√v + ε)   # 自适应学习率

        参数:
            params: dict, 模型参数
            grads: dict, 参数梯度
        """
        for k in params:
            # 首次遇到参数时初始化缓存 v
            if k not in self.cache:
                # np.zeros_like: 创建与参数同形的零数组
                self.cache[k] = np.zeros_like(params[k])

            # 加入权重衰减的梯度
            grad = grads[k] + self.weight_decay * params[k]

            # 更新梯度平方的移动平均
            # v = β·v + (1-β)·grad²
            # grad ** 2: 逐元素平方
            self.cache[k] = self.decay * self.cache[k] + (1 - self.decay) * (grad ** 2)

            # 自适应学习率更新: W = W - lr * grad / (√v + ε)
            # np.sqrt(self.cache[k]): 对 v 开方 (RMS)
            # grad / (√v + ε): 梯度除以 RMS（自适应缩放）
            params[k] -= self.lr * grad / (np.sqrt(self.cache[k]) + self.epsilon)
