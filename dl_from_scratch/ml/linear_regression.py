"""线性回归 — 最小二乘法 & 梯度下降"""

import numpy as np


class LinearRegression:
    """线性回归

    支持:
    - 闭式解 (OLS): θ = (X^T X)^(-1) X^T y
    - 梯度下降: 适用于大数据集

    Args:
        fit_intercept: 是否拟合截距项
        method: 'closed_form' 或 'gd' (梯度下降)
        lr: 学习率 (梯度下降时使用)
        epochs: 迭代次数 (梯度下降时使用)
    """

    def __init__(self, fit_intercept=True, method='closed_form', lr=0.01, epochs=1000):
        self.fit_intercept = fit_intercept
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = 0.0

    def _add_intercept(self, X):
        """添加偏置列"""
        if self.fit_intercept:
            return np.c_[np.ones(X.shape[0]), X]
        return X

    def fit(self, X, y):
        """训练模型

        Args:
            X: shape=(n_samples, n_features)
            y: shape=(n_samples,)
        """
        X_aug = self._add_intercept(X)
        n_samples, n_features = X_aug.shape

        if self.method == 'closed_form':
            # 正规方程: θ = (X^T X)^(-1) X^T y
            theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y

        elif self.method == 'gd':
            # 梯度下降
            theta = np.random.randn(n_features) * 0.01
            for _ in range(self.epochs):
                y_pred = X_aug @ theta
                gradient = (2.0 / n_samples) * X_aug.T @ (y_pred - y)
                theta -= self.lr * gradient
        else:
            raise ValueError(f"未知方法: {self.method}")

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta

        return self

    def predict(self, X):
        """预测

        Args:
            X: shape=(n_samples, n_features)

        Returns:
            y_pred: shape=(n_samples,)
        """
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R² 决定系数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        return f"LinearRegression(method={self.method})"
