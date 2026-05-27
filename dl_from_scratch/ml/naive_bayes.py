"""朴素贝叶斯 — 高斯朴素贝叶斯 (Gaussian Naive Bayes)

数学原理:
  贝叶斯定理: P(y|x) = P(y) * P(x|y) / P(x)
  朴素假设: 特征之间条件独立
    P(x|y) = P(x₁|y) * P(x₂|y) * ... * P(x_n|y)

  高斯朴素贝叶斯: 假设 P(x_i|y) 服从高斯分布 N(μ, σ²)

  预测时:
    ŷ = argmax_y log P(y) + Σ_i log P(x_i|y)
"""

import numpy as np


class GaussianNB:
    """高斯朴素贝叶斯分类器

    假设每个特征在每个类别下服从高斯分布（正态分布），
    特征之间条件独立。

    参数:
        无（参数在 fit 时从数据中学习）
    """

    def __init__(self):
        # classes_: 所有可能的类别标签
        self.classes_ = None
        # priors_: P(y)，每个类别的先验概率
        self.priors_ = None
        # means_: μ_{i,y}，每个类别下每个特征的均值
        # shape=(n_classes, n_features)
        self.means_ = None
        # variances_: σ_{i,y}²，每个类别下每个特征的方差
        # shape=(n_classes, n_features)
        self.variances_ = None

    def fit(self, X, y):
        """训练高斯朴素贝叶斯模型

        计算每个类别下每个特征的均值和方差（极大似然估计）。

        参数:
            X: ndarray, shape=(n_samples, n_features)
            y: ndarray, shape=(n_samples,)

        返回:
            self: 训练后的模型
        """
        # np.unique(y): 获取所有不同的类别标签
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        # n_classes: 类别总数
        n_classes = len(self.classes_)

        # 分配内存
        self.priors_ = np.zeros(n_classes)      # P(y)
        self.means_ = np.zeros((n_classes, n_features))    # μ
        self.variances_ = np.zeros((n_classes, n_features)) # σ²

        # 对每个类别分别计算统计量
        for i, c in enumerate(self.classes_):
            # X[y == c]: 取出所有属于类别 c 的样本（布尔索引）
            X_c = X[y == c]

            # 先验概率 P(y=c) = count(c) / total
            self.priors_[i] = len(X_c) / n_samples

            # 均值 μ_i = mean(X_c)，对每个特征分别计算
            # np.mean(X_c, axis=0): 在样本维度求平均 → (n_features,)
            self.means_[i] = np.mean(X_c, axis=0)

            # 方差 σ_i² = var(X_c)
            # np.var(X_c, axis=0): 在样本维度求方差 → (n_features,)
            # 1e-9: 加微小常数防止零方差（导致除零）
            self.variances_[i] = np.var(X_c, axis=0) + 1e-9

        return self

    def _log_likelihood(self, X):
        """计算对数似然 log P(X|y)

        对每个类别，计算每个样本在该类别下的对数概率密度。

        高斯分布的概率密度:
          P(x|y) = 1/(√(2πσ²)) * exp(-(x-μ)²/(2σ²))

        取对数后:
          log P(x|y) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            log_likelihood: ndarray, shape=(n_samples, n_classes)
                每个样本在每个类别下的对数似然
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        log_likelihood = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # diff = X - μ_i: 每个样本减去类别 i 的均值 → (n_samples, n_features)
            diff = X - self.means_[i]

            # 计算对数概率密度
            # np.log(2 * np.pi * self.variances_[i]): log(2πσ²)
            # (diff ** 2) / self.variances_[i]: (x-μ)²/σ²
            # np.sum(..., axis=1): 对特征维度求和（朴素贝叶斯的"朴素"假设）
            # -0.5 * sum: 完整的对数概率密度
            log_likelihood[:, i] = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances_[i]) +
                (diff ** 2) / self.variances_[i],
                axis=1
            )

        return log_likelihood

    def predict_proba(self, X):
        """预测后验概率 P(y|X)

        使用贝叶斯定理:
          P(y|X) ∝ P(y) * P(X|y) = P(y) * ∏ P(x_i|y)

        取对数避免浮点下溢:
          log P(y|X) = log P(y) + Σ log P(x_i|y) + constant

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            probs: ndarray, shape=(n_samples, n_classes)
                每个样本在每个类别下的概率（行和为 1）
        """
        # 对数先验: log P(y)
        # 1e-15: 防止 log(0)
        log_prior = np.log(self.priors_ + 1e-15)

        # 对数后验（未归一化）: log P(y) + log P(X|y)
        log_posterior = self._log_likelihood(X) + log_prior

        # ---- 数值稳定 softmax ----
        # 减去最大值防止 exp 溢出
        log_posterior -= np.max(log_posterior, axis=1, keepdims=True)
        exp_posterior = np.exp(log_posterior)
        # 归一化为概率
        return exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)

    def predict(self, X):
        """预测类别标签

        选择后验概率最大的类别。

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            y_pred: ndarray, shape=(n_samples,)
        """
        # np.argmax(predict_proba(X), axis=1): 概率最大的类别索引
        # self.classes_[...]: 将索引映射回原始标签值
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        """计算分类准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return "GaussianNB()"
