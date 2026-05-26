"""主成分分析 (PCA) — 降维"""

import numpy as np


class PCA:
    """主成分分析

    通过特征值分解协方差矩阵，找到数据的主要方向。

    Args:
        n_components: 保留的主成分数量
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None   # 主成分 (特征向量)
        self.explained_variance_ratio_ = None  # 方差解释比例
        self.mean_ = None
        self.singular_values_ = None

    def fit(self, X):
        """训练 PCA

        Args:
            X: shape=(n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 协方差矩阵
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 按特征值降序排列
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 数值稳定处理: 将微小负特征值置零
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 计算方差解释比例
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_var if total_var > 0 \
                                         else np.zeros_like(eigenvalues)

        # 奇异值 (用于变换)
        self.singular_values_ = np.sqrt(eigenvalues * (n_samples - 1))

        # 选择主成分
        if self.n_components is None:
            self.n_components = n_features
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

        return self

    def transform(self, X):
        """降维: X -> X_pca"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """训练并降维"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """逆变换: X_pca -> X (近似重建)"""
        return X_pca @ self.components_ + self.mean_

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"
