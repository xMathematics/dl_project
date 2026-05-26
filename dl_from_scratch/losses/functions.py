"""损失函数"""

import numpy as np


class MSELoss:
    """均方误差损失 (Mean Squared Error)

    L = 1/N * sum((y_pred - y_true)^2)
    """

    def forward(self, y_pred, y_true):
        """前向计算损失

        Args:
            y_pred: 模型预测, shape=(batch_size, ...)
            y_true: 真实标签, shape=(batch_size, ...)

        Returns:
            标量损失值
        """
        self.cache = (y_pred, y_true)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        """计算梯度

        Returns:
            dL/dy_pred, shape=(batch_size, ...)
        """
        y_pred, y_true = self.cache
        batch_size = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / batch_size

    def __repr__(self):
        return "MSELoss"


class CrossEntropyLoss:
    """交叉熵损失 (Cross Entropy) — 适用于多分类

    L = -1/N * sum(y_true * log(y_pred))
    其中 y_pred 是 softmax 输出的概率分布
    """

    def forward(self, y_pred, y_true):
        """前向计算损失

        Args:
            y_pred: 模型输出 (logits 或概率), shape=(batch_size, num_classes)
            y_true: 真实标签, shape=(batch_size,) 整数索引 或 (batch_size, num_classes) one-hot

        Returns:
            标量损失值
        """
        batch_size = y_pred.shape[0]

        # Softmax 数值稳定计算
        y_pred_max = np.max(y_pred, axis=-1, keepdims=True)
        y_pred_shifted = y_pred - y_pred_max
        y_pred_exp = np.exp(y_pred_shifted)
        y_pred_softmax = y_pred_exp / np.sum(y_pred_exp, axis=-1, keepdims=True)

        # 将 y_true 转换为 one-hot 编码 (如果需要)
        if y_true.ndim == 1:
            y_true_one_hot = np.zeros_like(y_pred_softmax)
            y_true_one_hot[np.arange(batch_size), y_true] = 1.0
        else:
            y_true_one_hot = y_true

        # 交叉熵
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred_softmax, eps, 1 - eps)
        loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred_clipped), axis=-1))

        # 缓存用于反向传播 (combined softmax + cross entropy gradient)
        self.cache = (y_pred_softmax, y_true_one_hot, batch_size)
        return loss

    def backward(self):
        """计算梯度 (Softmax + CrossEntropy 合并梯度)

        dL/dz = softmax(z) - y_true

        Returns:
            dL/dy_pred (logits 的梯度)
        """
        y_pred_softmax, y_true_one_hot, batch_size = self.cache
        return (y_pred_softmax - y_true_one_hot) / batch_size

    def __repr__(self):
        return "CrossEntropyLoss"


class BinaryCrossEntropyLoss:
    """二元交叉熵损失 — 适用于二分类

    输入 logits (网络最后一层的原始输出)，内部做 sigmoid + BCE。
    L = -1/N * sum(y * log(sigmoid(z)) + (1-y) * log(1-sigmoid(z)))
    """

    def forward(self, y_pred, y_true):
        """前向计算损失

        Args:
            y_pred: 模型输出 (logits), shape=(batch_size, 1) 或 (batch_size,)
            y_true: 真实标签, shape=(batch_size, 1) 或 (batch_size,)

        Returns:
            标量损失值
        """
        logits = y_pred
        # Sigmoid
        s = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
        eps = 1e-15
        s_clipped = np.clip(s, eps, 1 - eps)
        batch_size = logits.shape[0]

        loss = -np.mean(
            y_true * np.log(s_clipped) + (1 - y_true) * np.log(1 - s_clipped)
        )

        self.cache = (logits, y_true, batch_size)
        return loss

    def backward(self):
        """计算梯度 (Sigmoid + BCE 合并梯度)

        dL/dz = (sigmoid(z) - y) / batch_size
        """
        logits, y_true, batch_size = self.cache
        s = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))
        return (s - y_true) / batch_size

    def __repr__(self):
        return "BinaryCrossEntropyLoss"


class L1Loss:
    """L1 损失 (平均绝对误差)

    L = 1/N * sum(|y_pred - y_true|)
    """

    def forward(self, y_pred, y_true):
        self.cache = (y_pred, y_true)
        return np.mean(np.abs(y_pred - y_true))

    def backward(self):
        y_pred, y_true = self.cache
        batch_size = y_pred.shape[0]
        return np.sign(y_pred - y_true) / batch_size

    def __repr__(self):
        return "L1Loss"
