"""线性回归 — 最小二乘法 & 梯度下降

数学原理:
  y = X @ w + b
  优化目标: min_w ||Xw - y||²

两种求解方法:
  1. 闭式解 (正规方程): w = (X^T X)^(-1) X^T y
  2. 梯度下降: 迭代更新 w -= lr * ∇L
"""

# NumPy 是整个项目的唯一依赖，提供高效的数组运算和线性代数工具
import numpy as np


class LinearRegression:
    """线性回归模型

    支持两种求解方法:
    - closed_form: 使用正规方程直接求解（小数据集推荐）
    - gd: 梯度下降迭代求解（大数据集推荐）

    参数:
        fit_intercept: bool, 是否拟合截距项 b（默认 True）
        method: str, 'closed_form' 或 'gd'
        lr: float, 梯度下降的学习率
        epochs: int, 梯度下降的迭代次数
    """

    def __init__(self, fit_intercept=True, method='closed_form', lr=0.01, epochs=1000):
        # fit_intercept: 控制是否在特征矩阵前加一列 1（用于学习偏置项 b）
        self.fit_intercept = fit_intercept
        # method: 选择求解方式 — 'closed_form' 用正规方程，'gd' 用梯度下降
        self.method = method
        # lr (learning rate): 梯度下降的步长，太大不收敛，太小收敛慢
        self.lr = lr
        # epochs: 梯度下降遍历整个数据集的次数
        self.epochs = epochs
        # coef_: 模型权重系数 w，shape=(n_features,)，初始 None
        self.coef_ = None
        # intercept_: 截距项 b（偏置），初始为 0
        self.intercept_ = 0.0

    def _add_intercept(self, X):
        """在特征矩阵最左侧添加一列全 1，用于学习截距项

        np.c_[]: NumPy 的列拼接函数，将两个数组按列合并
        np.ones(X.shape[0]): 创建长度为样本数的全 1 向量
        效果: X 从 (n, d) 变为 (n, d+1)，第一列全为 1
        """
        if self.fit_intercept:
            # np.c_ 是 NumPy 的列连接器，等价于 np.concatenate(axis=1)
            # 添加全 1 列后，线性模型变为: y = w0*1 + w1*x1 + w2*x2 + ...
            # 其中 w0 就是截距项 intercept
            return np.c_[np.ones(X.shape[0]), X]
        return X

    def fit(self, X, y):
        """训练线性回归模型

        根据 self.method 选择不同的求解算法。

        参数:
            X: ndarray, shape=(n_samples, n_features)，特征矩阵
            y: ndarray, shape=(n_samples,)，目标值（连续实数）

        返回:
            self: 训练后的模型实例
        """
        # 添加偏置列: X_aug 比 X 多一列全 1
        X_aug = self._add_intercept(X)
        # X_aug.shape[0]: 样本数 n_samples
        # X_aug.shape[1]: 特征数 n_features（含偏置列）
        n_samples, n_features = X_aug.shape

        if self.method == 'closed_form':
            # === 方法1: 正规方程 (Normal Equation) ===
            # 公式: θ = (X^T X)^(-1) X^T y
            # 这是最小二乘法的闭式解，一步到位求出最优参数
            #
            # np.linalg.pinv(): 伪逆矩阵 (Moore-Penrose)，比 inv() 更稳定
            #   当 X^T X 不可逆时（特征相关），pinv 仍能工作
            # X_aug.T @ X_aug: 计算 X^T X，@ 是 NumPy 的矩阵乘法运算符
            # @ X_aug.T @ y: 完整的正规方程计算
            theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y

        elif self.method == 'gd':
            # === 方法2: 梯度下降 (Gradient Descent) ===
            # 随机初始化参数: np.random.randn 生成标准正态分布随机数
            # * 0.01: 缩小初始值范围，防止梯度爆炸
            theta = np.random.randn(n_features) * 0.01

            # 迭代优化: 循环 epochs 次
            for _ in range(self.epochs):
                # 前向传播: 计算预测值 y_pred = X @ theta
                # @ 是矩阵乘法: (n, d) @ (d,) -> (n,)
                y_pred = X_aug @ theta

                # 计算梯度: ∇L = (2/n) * X^T (y_pred - y)
                # (2.0 / n_samples): 损失函数 MSE = (1/n)Σ(y_pred-y)² 的导数系数
                # X_aug.T @ (y_pred - y): X^T 乘以误差向量
                gradient = (2.0 / n_samples) * X_aug.T @ (y_pred - y)

                # 参数更新: θ = θ - lr * ∇L
                # 沿梯度反方向移动，逐步逼近最优解
                theta -= self.lr * gradient
        else:
            # 未知方法名，抛出 ValueError 异常
            raise ValueError(f"未知方法: {self.method}")

        # 解析参数: 如果加了偏置列，第一个参数是截距
        if self.fit_intercept:
            # theta[0]: 对应全 1 列的权重，即截距 b
            self.intercept_ = theta[0]
            # theta[1:]: 剩余的参数是各特征的权重系数 w
            self.coef_ = theta[1:]
        else:
            # 没有偏置列，所有参数都是特征权重
            self.coef_ = theta

        # 返回 self 以支持链式调用 (如 model.fit(X,y).predict(X))
        return self

    def predict(self, X):
        """预测: y = X @ w + b

        线性回归的预测就是简单的矩阵乘法和加法。

        参数:
            X: ndarray, shape=(n_samples, n_features)，待预测的特征矩阵

        返回:
            y_pred: ndarray, shape=(n_samples,)，预测值
        """
        # X @ self.coef_: 矩阵乘法，每个样本的特征乘以对应权重后求和
        # + self.intercept_: 加上截距项（偏置）
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """计算 R² 决定系数 (Coefficient of Determination)

        R² = 1 - SS_res / SS_tot
        - SS_res = Σ(y - y_pred)²: 残差平方和，模型未能解释的方差
        - SS_tot = Σ(y - y_mean)²: 总平方和，数据的总体方差
        - R² = 1: 完美拟合 | R² = 0: 等价于均值预测 | R² < 0: 比均值预测还差

        参数:
            X: ndarray, 特征矩阵
            y: ndarray, 真实目标值

        返回:
            r2: float, R² 值 (最大为 1，可能为负)
        """
        # 获取预测值
        y_pred = self.predict(X)
        # SS_res: 残差平方和，np.sum 对数组所有元素求和
        # (y - y_pred) ** 2: 逐元素相减后平方（NumPy 广播机制）
        ss_res = np.sum((y - y_pred) ** 2)
        # SS_tot: 总平方和，np.mean(y) 计算 y 的均值
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # 1e-15: 防止除零的微小常数（epsilon）
        return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        """返回模型的字符串表示，方便调试"""
        return f"LinearRegression(method={self.method})"
