"""K 均值聚类 (K-Means) — 无监督学习"""

import numpy as np


class KMeans:
    """K 均值聚类

    原理: 迭代地将样本分配到最近的聚类中心，然后更新中心。

    Args:
        n_clusters: 聚类数 K
        max_iter: 最大迭代次数
        tol: 收敛阈值 (中心变化小于此值时停止)
        random_seed: 随机种子
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_seed=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed = random_seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # 样本到最近中心的距离平方和

    def _init_centroids(self, X):
        """K-Means++ 初始化 — 选择相距较远的初始中心"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_samples = X.shape[0]
        centroids = []

        # 随机选择第一个中心
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.n_clusters):
            # 计算每个样本到最近中心的距离
            dists = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in X])
            # 按距离加权概率选择下一个中心
            probs = dists / np.sum(dists)
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X):
        """训练 K-Means

        Args:
            X: shape=(n_samples, n_features)

        Returns:
            self
        """
        n_samples = X.shape[0]

        # 初始化中心
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # 分配步骤: 每个样本分配到最近的中心
            distances = self._compute_distances(X)  # (n_samples, n_clusters)
            labels = np.argmin(distances, axis=1)

            # 更新步骤: 重新计算中心
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    new_centroids[k] = np.mean(X[labels == k], axis=0)
                else:
                    new_centroids[k] = self.centroids[k]  # 保持原中心

            # 检查收敛
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        # 最终分配
        distances = self._compute_distances(X)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)

        return self

    def _compute_distances(self, X):
        """计算每个样本到每个中心的距离"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - self.centroids[k]
            distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances

    def predict(self, X):
        """预测每个样本所属的聚类"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def transform(self, X):
        """将样本转换到聚类中心距离空间"""
        return self._compute_distances(X)

    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return f"KMeans(n_clusters={self.n_clusters})"
