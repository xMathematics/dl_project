"""朴素贝叶斯 — 高斯朴素贝叶斯 (Gaussian Naive Bayes)"""

import numpy as np


class GaussianNB:
    """高斯朴素贝叶斯分类器

    假设每个特征服从高斯分布，特征之间条件独立。

    P(y|x) ∝ P(y) * ∏ P(x_i | y)
    P(x_i | y) = N(x_i | μ_{i,y}, σ_{i,y}²)
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None      # P(y)
        self.means_ = None       # μ_{i,y}
        self.variances_ = None   # σ_{i,y}²

    def fit(self, X, y):
        """训练模型

        Args:
            X: shape=(n_samples, n_features)
            y: shape=(n_samples,)
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = len(X_c) / n_samples
            self.means_[i] = np.mean(X_c, axis=0)
            self.variances_[i] = np.var(X_c, axis=0) + 1e-9  # 防止除零

        return self

    def _log_likelihood(self, X):
        """计算对数似然 log P(X|y)

        多元高斯对数概率密度:
        log P(x|y) = -0.5 * (log(2πσ²) + (x-μ)²/σ²)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        log_likelihood = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            diff = X - self.means_[i]
            log_likelihood[:, i] = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances_[i]) +
                (diff ** 2) / self.variances_[i],
                axis=1
            )

        return log_likelihood

    def predict_proba(self, X):
        """预测概率 P(y|X)"""
        log_prior = np.log(self.priors_ + 1e-15)
        log_posterior = self._log_likelihood(X) + log_prior

        # Softmax 将 log 概率转换为概率
        log_posterior -= np.max(log_posterior, axis=1, keepdims=True)
        exp_posterior = np.exp(log_posterior)
        return exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)

    def predict(self, X):
        """预测类别"""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        """准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return "GaussianNB()"
