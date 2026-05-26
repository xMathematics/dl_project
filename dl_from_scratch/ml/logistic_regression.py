"""逻辑回归 — 二分类 (梯度下降)"""

import numpy as np


class LogisticRegression:
    """逻辑回归

    y = sigmoid(X @ w + b)
    使用交叉熵损失 + 梯度下降训练。

    Args:
        lr: 学习率
        epochs: 迭代次数
        fit_intercept: 是否拟合偏置
    """

    def __init__(self, lr=0.1, epochs=1000, fit_intercept=True):
        self.lr = lr
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))

    def fit(self, X, y):
        """训练模型

        Args:
            X: shape=(n_samples, n_features)
            y: shape=(n_samples,), 取值 {0, 1}
        """
        n_samples, n_features = X.shape
        y = y.astype(float)

        # 初始化参数
        if self.fit_intercept:
            self.coef_ = np.random.randn(n_features) * 0.01
            self.intercept_ = 0.0
        else:
            self.coef_ = np.random.randn(n_features) * 0.01

        for epoch in range(self.epochs):
            # 线性部分
            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)

            # 交叉熵损失
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

            # 梯度
            error = y_pred - y  # (sigmoid(z) - y)
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # 更新
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

            # 学习率衰减
            if epoch % 200 == 0 and epoch > 0:
                self.lr *= 0.95

        return self

    def predict_proba(self, X):
        """预测概率"""
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"LogisticRegression(lr={self.lr}, epochs={self.epochs})"
