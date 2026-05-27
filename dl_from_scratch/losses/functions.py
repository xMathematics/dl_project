"""损失函数

损失函数衡量模型预测值与真实值之间的差异。
训练的目标就是最小化损失函数的值。

本模块实现了三种损失函数:
- MSELoss: 回归任务
- CrossEntropyLoss: 多分类任务 (内部集成 Softmax)
- BinaryCrossEntropyLoss: 二分类任务 (内部集成 Sigmoid)
- L1Loss: 回归任务 (鲁棒性更强)

注意: CrossEntropy/BCE 接收 logits (原始输出)，内部处理激活函数。
"""

import numpy as np


class MSELoss:
    """均方误差损失 (Mean Squared Error)

    用于回归任务。

    数学公式:
      L = (1/N) * Σ(y_pred - y_true)²

    梯度:
      dL/d(y_pred) = 2 * (y_pred - y_true) / N
    """

    def forward(self, y_pred, y_true):
        """前向计算 MSE 损失

        np.mean((y_pred - y_true) ** 2): 先逐元素平方，再求均值

        参数:
            y_pred: ndarray, 模型预测值
            y_true: ndarray, 真实值

        返回:
            loss: float, 标量损失值
        """
        # 缓存预测和真实值，供 backward 使用
        self.cache = (y_pred, y_true)
        # (y_pred - y_true) ** 2: 逐元素平方差
        # np.mean(): 对所有元素求平均
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        """计算 MSE 梯度

        公式: dL/d(y_pred) = 2 * (y_pred - y_true) / N

        返回:
            grad: ndarray, 与 y_pred 形状相同
        """
        y_pred, y_true = self.cache
        # y_pred.shape[0]: batch 大小 N
        batch_size = y_pred.shape[0]
        # 2 * (y_pred - y_true) / N: MSE 的导数
        return 2.0 * (y_pred - y_true) / batch_size

    def __repr__(self):
        return "MSELoss"


class CrossEntropyLoss:
    """交叉熵损失 (Cross Entropy) — 多分类

    内部集成了 Softmax 函数，所以网络输出层不需要加 Softmax。
    直接传入 logits (原始输出) 即可。

    数学公式:
      L = -(1/N) * Σ Σ y_true * log(softmax(y_pred))

    梯度 (Softmax + CrossEntropy 合并):
      dL/dz = (softmax(z) - y_true) / N
      这个简洁形式是交叉熵和 softmax 组合的巧妙结果！
    """

    def forward(self, y_pred, y_true):
        """前向计算交叉熵损失

        步骤:
        1. 对 y_pred (logits) 做数值稳定的 Softmax
        2. 将 y_true 转为 one-hot 编码 (如果是整数标签)
        3. 计算交叉熵

        参数:
            y_pred: ndarray, shape=(batch_size, num_classes), logits
            y_true: ndarray, shape=(batch_size,) 整数标签,
                    或 (batch_size, num_classes) one-hot

        返回:
            loss: float, 标量损失值
        """
        batch_size = y_pred.shape[0]

        # ---- 数值稳定的 Softmax ----
        # 技巧: 减去最大值，防止 exp 上溢出
        # np.max(y_pred, axis=-1, keepdims=True): 每行的最大值，保持维度
        y_pred_max = np.max(y_pred, axis=-1, keepdims=True)
        y_pred_shifted = y_pred - y_pred_max  # 最大值为 0
        y_pred_exp = np.exp(y_pred_shifted)   # 最大值为 e^0 = 1
        # 归一化: e^(z_i) / Σe^(z_j)
        y_pred_softmax = y_pred_exp / np.sum(y_pred_exp, axis=-1, keepdims=True)

        # ---- 标签转 one-hot ----
        if y_true.ndim == 1:
            # 整数标签 -> one-hot 编码
            # np.zeros_like(y_pred_softmax): 创建与 softmax 输出同形的全零矩阵
            y_true_one_hot = np.zeros_like(y_pred_softmax)
            # np.arange(batch_size): [0, 1, 2, ..., N-1]
            # y_true: 每个样本的真实类别索引
            # 将对应位置置为 1
            y_true_one_hot[np.arange(batch_size), y_true] = 1.0
        else:
            y_true_one_hot = y_true

        # ---- 计算交叉熵 ----
        # L = -Σ y_true * log(softmax(z))
        # np.clip(softmax, eps, 1-eps): 防止 log(0) 导致 -inf
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_softmax, eps, 1 - eps)

        # y_true_one_hot * np.log(y_pred_clipped): 只取真实类别对应的 log 概率
        # np.sum(..., axis=-1): 对类别维度求和
        # np.mean(): 对所有样本取平均
        loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_clipped), axis=-1))

        # 缓存用于反向传播
        self.cache = (y_pred_softmax, y_true_one_hot, batch_size)
        return loss

    def backward(self):
        """计算梯度 (Softmax + CrossEntropy 合并)

        公式: dL/dz = (softmax(z) - y_true) / N

        这合并了 softmax 的雅可比矩阵和交叉熵的梯度，
        推导过程: softmax(z) -> s, CE(s) -> -ylog(s)
        dL/dz = s * (dCE/ds - sum(dCE/ds * s)) = (s - y) / N

        返回:
            grad: ndarray, shape=(batch_size, num_classes), logits 的梯度
        """
        y_pred_softmax, y_true_one_hot, batch_size = self.cache
        # (softmax - y_true) / batch_size: 合并梯度
        return (y_pred_softmax - y_true_one_hot) / batch_size

    def __repr__(self):
        return "CrossEntropyLoss"


class BinaryCrossEntropyLoss:
    """二元交叉熵损失 — 二分类

    内部集成了 Sigmoid 函数，网络输出层不需要加 Sigmoid。
    直接传入 logits (原始输出) 即可。

    数学公式:
      L = -(1/N) * Σ [y * log(σ(z)) + (1-y) * log(1-σ(z))]

    梯度 (Sigmoid + BCE 合并):
      dL/dz = (σ(z) - y) / N

    其中 σ(z) = sigmoid(z) = 1/(1+e^{-z})
    """

    def forward(self, y_pred, y_true):
        """前向计算二元交叉熵损失

        步骤:
        1. 对 logits 做 sigmoid 得到概率 s = σ(z)
        2. 计算 BCE = -[y·log(s) + (1-y)·log(1-s)]

        参数:
            y_pred: ndarray, logits (任意实数，网络最后一层的原始输出)
            y_true: ndarray, 标签, 取值 {0, 1}, shape=(batch_size, 1)

        返回:
            loss: float, 标量损失值
        """
        logits = y_pred

        # ---- Sigmoid: σ(z) = 1 / (1 + e^{-z}) ----
        # np.clip(logits, -100, 100): 防止 exp 溢出
        s = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))

        # ---- 数值稳定处理 ----
        eps = 1e-15
        # np.clip(s, eps, 1 - eps): 防止 log(0)
        s_clipped = np.clip(s, eps, 1 - eps)

        batch_size = logits.shape[0]

        # ---- 计算 BCE ----
        # y_true * np.log(s) + (1-y_true) * log(1-s): BCE 公式
        # np.mean(): 对所有样本取平均
        loss = -np.mean(
            y_true * np.log(s_clipped) + (1 - y_true) * np.log(1 - s_clipped)
        )

        # 缓存用于反向传播
        self.cache = (logits, y_true, batch_size)
        return loss

    def backward(self):
        """计算梯度 (Sigmoid + BCE 合并)

        公式: dL/dz = (σ(z) - y) / N

        推导: dL/dz = dL/dσ * dσ/dz
             dL/dσ = -(y/σ - (1-y)/(1-σ)) = (σ-y)/(σ(1-σ))
             dσ/dz = σ(1-σ)
             所以 dL/dz = (σ-y)/N

        返回:
            grad: ndarray, shape=(batch_size, 1), logits 的梯度
        """
        logits, y_true, batch_size = self.cache
        # 重新计算 sigmoid
        s = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
        # 合并梯度: (s - y) / N
        return (s - y_true) / batch_size

    def __repr__(self):
        return "BinaryCrossEntropyLoss"


class L1Loss:
    """L1 损失 (平均绝对误差 / Mean Absolute Error)

    用于回归任务，相比 MSE 对异常值更鲁棒。

    数学公式:
      L = (1/N) * Σ |y_pred - y_true|

    梯度:
      dL/d(y_pred) = sign(y_pred - y_true) / N
    """

    def forward(self, y_pred, y_true):
        """前向计算 L1 损失

        np.abs(y_pred - y_true): 逐元素计算绝对值
        np.mean(...): 对所有元素求平均

        参数:
            y_pred: ndarray, 模型预测值
            y_true: ndarray, 真实值

        返回:
            loss: float, 标量损失值
        """
        # 缓存用于反向传播
        self.cache = (y_pred, y_true)
        # L1 = mean(|y_pred - y_true|)
        return np.mean(np.abs(y_pred - y_true))

    def backward(self):
        """计算 L1 梯度

        np.sign(x): 符号函数
          x > 0  → 1
          x < 0  → -1
          x = 0  → 0

        返回:
            grad: ndarray, 与 y_pred 形状相同
        """
        y_pred, y_true = self.cache
        batch_size = y_pred.shape[0]
        # dL/d(y_pred) = sign(y_pred - y_true) / N
        return np.sign(y_pred - y_true) / batch_size

    def __repr__(self):
        return "L1Loss"
