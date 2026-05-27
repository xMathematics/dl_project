"""逻辑回归 — 二分类 (梯度下降实现)

数学原理:
  y = sigmoid(X @ w + b)
  sigmoid(z) = 1 / (1 + e^(-z))  # 将任意实数映射到 (0,1) 区间

损失函数: 二元交叉熵 (Binary Cross-Entropy)
  L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

梯度 (关键推导):
  ∂L/∂w = (sigmoid(Xw+b) - y) · X  / n
  ∂L/∂b = mean(sigmoid(Xw+b) - y)
"""

import numpy as np


class LogisticRegression:
    """逻辑回归二分类器

    使用梯度下降优化交叉熵损失，支持学习率衰减。

    参数:
        lr: float, 初始学习率
        epochs: int, 梯度下降迭代次数
        fit_intercept: bool, 是否拟合偏置项
    """

    def __init__(self, lr=0.1, epochs=1000, fit_intercept=True):
        # lr: 学习率 (learning rate)，控制参数更新的步长
        self.lr = lr
        # epochs: 完整遍历训练数据的次数
        self.epochs = epochs
        # fit_intercept: 是否学习偏置项 b（建议为 True）
        self.fit_intercept = fit_intercept
        # coef_: 特征权重 w，训练后 shape=(n_features,)
        self.coef_ = None
        # intercept_: 偏置（截距）b
        self.intercept_ = 0.0

    def _sigmoid(self, z):
        """Sigmoid 激活函数

        σ(z) = 1 / (1 + e^(-z))
        将任意实数 z 压缩到 (0, 1) 区间，输出可解释为概率 P(y=1|x)

        np.exp(z): NumPy 的指数函数 e^z，对数组每个元素计算
        np.clip(z, -100, 100): 将 z 限制在 [-100, 100] 区间
            防止 exp(-100) = 3.7e-44 下溢出为零，或 exp(100) 上溢出为 inf
        """
        # 数值稳定版 sigmoid: 先 clip 再 exp，避免溢出
        # exp(-z) 当 z 很大时接近 0，当 z 很小时接近无穷
        return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))

    def fit(self, X, y):
        """训练逻辑回归模型

        使用梯度下降优化，每 200 轮学习率衰减 5%。

        参数:
            X: ndarray, shape=(n_samples, n_features)，特征矩阵
            y: ndarray, shape=(n_samples,)，标签值，取值 {0, 1}

        返回:
            self: 训练后的模型实例
        """
        # n_samples: 样本数，X.shape[0]
        # n_features: 特征数，X.shape[1]
        n_samples, n_features = X.shape
        # 确保 y 是 float 类型，避免整数运算导致的精度问题
        y = y.astype(float)

        # ---- 参数初始化 ----
        if self.fit_intercept:
            # np.random.randn(n_features): 生成 n_features 个标准正态分布 N(0,1) 随机数
            # * 0.01: 缩小到 0.01 量级，避免初始值过大导致 sigmoid 饱和
            self.coef_ = np.random.randn(n_features) * 0.01
            # 偏置初始化为 0
            self.intercept_ = 0.0
        else:
            # 不拟合偏置，只初始化权重
            self.coef_ = np.random.randn(n_features) * 0.01

        # ---- 梯度下降迭代 ----
        for epoch in range(self.epochs):
            # --- 前向传播 ---
            # z = X @ w + b: 线性部分，@ 是矩阵乘法
            # 结果 z 是每个样本的 logit（对数几率）
            z = X @ self.coef_ + self.intercept_
            # ŷ = sigmoid(z): 将 logit 转换为概率
            y_pred = self._sigmoid(z)

            # --- 计算损失 (仅用于监控，不参与梯度计算) ---
            # 二元交叉熵: -1/n * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
            # np.log: 自然对数
            # 1e-15: 防止 log(0) 导致 -inf
            loss = -np.mean(y * np.log(y_pred + 1e-15) +
                            (1 - y) * np.log(1 - y_pred + 1e-15))

            # --- 反向传播 (计算梯度) ---
            # error = ŷ - y: 预测误差，Sigmoid+BCE 的合并梯度
            # 这个简洁的形式是交叉熵和 sigmoid 组合的妙处
            error = y_pred - y

            # dw = X^T @ error / n: 权重的梯度
            # X.T: 矩阵转置 (n_features, n_samples)
            # X.T @ error: (n_features, n_samples) @ (n_samples,) -> (n_features,)
            dw = (X.T @ error) / n_samples

            # db = mean(error): 偏置的梯度
            # np.mean: 对所有样本的误差求平均
            db = np.mean(error)

            # --- 参数更新 ---
            # w = w - lr * dw: 沿梯度反方向更新
            self.coef_ -= self.lr * dw
            # b = b - lr * db: 更新偏置
            self.intercept_ -= self.lr * db

            # --- 学习率衰减 ---
            # 每 200 轮将学习率乘以 0.95，后期精细调整
            # 前期大步搜索，后期小步收敛
            if epoch % 200 == 0 and epoch > 0:
                self.lr *= 0.95

        return self

    def predict_proba(self, X):
        """预测样本属于类别 1 的概率 P(y=1|x)

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            probs: ndarray, shape=(n_samples,)，每个样本的概率值
        """
        # 计算 logits: z = X @ w + b
        z = X @ self.coef_ + self.intercept_
        # sigmoid 将 logits 转换为 (0,1) 区间的概率
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别标签 {0, 1}

        参数:
            X: ndarray, shape=(n_samples, n_features)
            threshold: float, 决策阈值（默认 0.5）

        返回:
            labels: ndarray, shape=(n_samples,)，值为 0 或 1
        """
        # predict_proba >= threshold: 概率超过阈值则为正类
        # .astype(int): 将布尔值 True/False 转为 1/0
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """计算分类准确率 (Accuracy)

        accuracy = 预测正确的样本数 / 总样本数

        参数:
            X: ndarray, 特征矩阵
            y: ndarray, 真实标签

        返回:
            acc: float, 准确率 (0~1)
        """
        # np.mean(predict(X) == y): 先比较预测和真实值（返回布尔数组），再求平均
        # True=1, False=0，均值就是准确率
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        """返回模型的字符串表示"""
        return f"LogisticRegression(lr={self.lr}, epochs={self.epochs})"
