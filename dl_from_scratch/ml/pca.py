"""主成分分析 (PCA) — 降维

数学原理:
  PCA 通过线性变换将高维数据映射到低维空间，保留数据的主要方差。

  步骤:
  1. 数据中心化: X_centered = X - mean(X)
  2. 计算协方差矩阵: Σ = (1/n) * X_centered^T @ X_centered
  3. 特征值分解: Σ = V * Λ * V^T
  4. 选择前 k 个最大特征值对应的特征向量作为主成分

  方差解释比例 = λ_i / Σλ_j: 每个主成分保留的信息量
"""

import numpy as np


class PCA:
    """主成分分析 (PCA) 降维模型

    通过特征值分解找到数据方差最大的方向（主成分），
    将高维数据投影到低维空间。

    参数:
        n_components: int, 保留的主成分数量（默认 None=保留全部）
    """

    def __init__(self, n_components=None):
        # n_components: 降维后的维度数
        self.n_components = n_components
        # components_: 主成分向量，shape=(n_components, n_features)
        # 每行是一个主成分方向（单位向量）
        self.components_ = None
        # explained_variance_ratio_: 各主成分的方差解释比例
        # 越大的值表示该主成分越重要
        self.explained_variance_ratio_ = None
        # mean_: 训练数据的均值向量，用于中心化
        self.mean_ = None
        # singular_values_: 奇异值 = sqrt(λ * (n-1))
        self.singular_values_ = None

    def fit(self, X):
        """训练 PCA 模型

        计算数据的协方差矩阵并进行特征值分解。

        参数:
            X: ndarray, shape=(n_samples, n_features)，输入数据

        返回:
            self: 训练后的模型
        """
        n_samples, n_features = X.shape

        # ---- 步骤1: 数据中心化 ----
        # np.mean(X, axis=0): 对每个特征求均值，shape=(n_features,)
        self.mean_ = np.mean(X, axis=0)
        # X - self.mean_: 广播减法，每个样本减去均值向量
        # 中心化后数据的均值为 0
        X_centered = X - self.mean_

        # ---- 步骤2: 计算协方差矩阵 ----
        # X_centered.T @ X_centered: (n_features, n_samples) @ (n_samples, n_features)
        #   → (n_features, n_features)，这是"散布矩阵"
        # / (n_samples - 1): 除以 n-1 得到无偏协方差估计
        #   (n_samples - 1 是自由度校正)
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # ---- 步骤3: 特征值分解 ----
        # np.linalg.eigh(cov): 对称矩阵的特征值分解
        #   eigenvalues: 特征值，按升序排列
        #   eigenvectors: 特征向量矩阵，每列是一个特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 特征值默认升序排列，降序排列（最大特征值在前）
        # np.argsort(eigenvalues)[::-1]: 降序排列的索引
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # ---- 数值稳定处理 ----
        # 由于浮点误差，部分特征值可能为极小的负数
        # np.maximum(eigenvalues, 0.0): 将负特征值置零
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # ---- 计算方差解释比例 ----
        total_var = np.sum(eigenvalues)
        # 每个特征值占总特征值之和的比例
        self.explained_variance_ratio_ = (
            eigenvalues / total_var if total_var > 0
            else np.zeros_like(eigenvalues)
        )

        # ---- 奇异值 ----
        # 奇异值 = sqrt(λ * (n-1))，用于 SVD 风格的变换
        self.singular_values_ = np.sqrt(eigenvalues * (n_samples - 1))

        # ---- 选择主成分 ----
        if self.n_components is None:
            self.n_components = n_features
        # eigenvectors[:, :self.n_components]: 取前 k 个特征向量
        # .T: 转置为 (n_components, n_features)，每行一个主成分
        self.components_ = eigenvectors[:, :self.n_components].T
        # 只保留前 k 个方差解释比例
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

        return self

    def transform(self, X):
        """将数据降维到主成分空间

        X_pca = (X - mean) @ components^T

        参数:
            X: ndarray, shape=(n_samples, n_features)，原始数据

        返回:
            X_pca: ndarray, shape=(n_samples, n_components)，降维后数据
        """
        # 中心化
        X_centered = X - self.mean_
        # X_centered @ self.components_.T: 投影到主成分空间
        # (n_samples, n_features) @ (n_features, n_components) -> (n_samples, n_components)
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """训练并降维，一步完成"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """逆变换: 从主成分空间重建原始数据

        X_reconst = X_pca @ components + mean

        由于降维丢弃了部分信息，重建是近似值。

        参数:
            X_pca: ndarray, shape=(n_samples, n_components)，主成分空间数据

        返回:
            X_reconst: ndarray, shape=(n_samples, n_features)，重建数据
        """
        # X_pca @ self.components_: (n_samples, n_components) @ (n_components, n_features)
        # + self.mean_: 加回均值
        return X_pca @ self.components_ + self.mean_

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"
