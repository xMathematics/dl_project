"""K 均值聚类 (K-Means) — 无监督学习

算法原理:
  K-Means 将数据划分为 K 个簇，使簇内样本相似度最高、簇间差异最大。

  迭代步骤 (EM 风格):
  1. 分配 (E-step): 将每个样本分配到最近的聚类中心
  2. 更新 (M-step): 重新计算每个簇的中心（均值）
  3. 重复直到收敛

  初始化: K-Means++ 选择相距较远的初始中心，提高收敛质量

  评估指标: inertia = Σ(每个样本到其簇中心的距离²)
"""

import numpy as np


class KMeans:
    """K 均值聚类模型

    使用 K-Means++ 初始化，支持自动收敛判断。

    参数:
        n_clusters: int, 聚类数量 K
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值（中心移动平方和小于此值时停止）
        random_seed: int or None, 随机种子（用于结果复现）
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_seed=None):
        # n_clusters: 要将数据分成多少个簇
        self.n_clusters = n_clusters
        # max_iter: 最大迭代次数，防止无限循环
        self.max_iter = max_iter
        # tol (tolerance): 收敛容差，中心点变化小于此值时算法停止
        self.tol = tol
        # random_seed: 随机种子，固定后可复现结果
        self.random_seed = random_seed
        # centroids: 聚类中心点，shape=(n_clusters, n_features)
        self.centroids = None
        # labels_: 每个样本的聚类分配结果，shape=(n_samples,)
        self.labels_ = None
        # inertia_: 簇内平方和 (Within-Cluster Sum of Squares)
        # 衡量聚类紧密度，越小说明簇越紧密
        self.inertia_ = None

    def _init_centroids(self, X):
        """K-Means++ 初始化策略

        基本 K-Means 随机初始化可能导致次优解，K-Means++ 改进为:
        1. 随机选第一个中心
        2. 后续中心以正比于距离的概率从远处选择
        3. 显著提高收敛质量和速度

        参数:
            X: ndarray, shape=(n_samples, n_features)，输入数据

        返回:
            centroids: ndarray, shape=(n_clusters, n_features)，初始中心
        """
        if self.random_seed is not None:
            # np.random.seed(): 设置随机数种子，使结果可复现
            np.random.seed(self.random_seed)

        # n_samples = X.shape[0]: 样本总数
        n_samples = X.shape[0]
        centroids = []

        # ---- 第一步: 随机选择第一个中心 ----
        # np.random.randint(n_samples): 从 [0, n_samples) 随机选一个整数
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        # ---- 第二步: 迭代选择剩余中心 ----
        for _ in range(1, self.n_clusters):
            # 对每个样本，计算它到所有已选中心的最近距离
            # min(np.linalg.norm(x - c) for c in centroids): 到最近中心的距离
            # np.linalg.norm(): 计算 L2 范数（欧氏距离）
            dists = np.array([
                min(np.linalg.norm(x - c) for c in centroids)
                for x in X
            ])

            # 按距离加权概率分布: 距离越远的样本被选中的概率越大
            # dists / np.sum(dists): 归一化为概率（和为 1）
            probs = dists / np.sum(dists)

            # np.random.choice(n_samples, p=probs): 按指定概率分布随机采样
            next_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        # 将列表转换为 NumPy 数组返回
        return np.array(centroids)

    def fit(self, X):
        """训练 K-Means 模型

        迭代执行"分配-更新"直到收敛或达到最大迭代次数。

        参数:
            X: ndarray, shape=(n_samples, n_features)，输入数据

        返回:
            self: 训练后的模型实例
        """
        # n_samples = X.shape[0]: 样本数
        n_samples = X.shape[0]

        # ---- 初始化聚类中心 (K-Means++) ----
        self.centroids = self._init_centroids(X)

        # ---- EM 风格迭代 ----
        for iteration in range(self.max_iter):
            # --- E-step: 分配每个样本到最近的中心 ---
            # _compute_distances(X): 计算距离矩阵 (n_samples, n_clusters)
            distances = self._compute_distances(X)
            # np.argmin(distances, axis=1): 对每个样本，找最近中心的索引
            # axis=1: 在列维度（各中心）上找最小值
            labels = np.argmin(distances, axis=1)

            # --- M-step: 重新计算每个簇的中心 ---
            # np.zeros_like(self.centroids): 创建与 centroids 相同形状的零数组
            new_centroids = np.zeros_like(self.centroids)

            for k in range(self.n_clusters):
                # labels == k: 布尔掩码，标记属于第 k 簇的样本
                if np.sum(labels == k) > 0:
                    # np.mean(X[labels == k], axis=0): 对第 k 簇所有样本求均值
                    # axis=0: 对样本维度求平均，得到新的中心坐标
                    new_centroids[k] = np.mean(X[labels == k], axis=0)
                else:
                    # 如果某簇没有样本（空簇），保持原中心不变
                    new_centroids[k] = self.centroids[k]

            # --- 检查收敛 ---
            # shift = Σ(新旧中心差的平方): 衡量中心移动量
            # np.sum(... ** 2): 先逐元素平方，再求和
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids

            # 如果中心移动小于容差，认为已收敛，提前退出
            if shift < self.tol:
                break

        # ---- 最终分配和评价 ----
        distances = self._compute_distances(X)
        # labels_: 每个样本的最终簇分配
        self.labels_ = np.argmin(distances, axis=1)
        # inertia_: 簇内平方和，衡量聚类质量
        # np.min(distances, axis=1): 每个样本到其最近中心的距离
        # ** 2: 平方后求和
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)

        return self

    def _compute_distances(self, X):
        """计算每个样本到每个聚类中心的欧氏距离

        distances[i, k] = ||X[i] - centroids[k]||₂

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            distances: ndarray, shape=(n_samples, n_clusters)
        """
        # n_samples = X.shape[0]
        n_samples = X.shape[0]
        # 初始化距离矩阵，全零 (n_samples, n_clusters)
        distances = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            # diff = X - centroids[k]: 广播相减，每个样本减去中心 k
            diff = X - self.centroids[k]
            # np.sqrt(np.sum(diff ** 2, axis=1)): 计算欧氏距离
            # diff ** 2: 逐元素平方
            # np.sum(..., axis=1): 对特征维度求和 -> (n_samples,)
            # np.sqrt(): 开平方 -> 欧氏距离
            distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))

        return distances

    def predict(self, X):
        """预测每个样本所属的聚类

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            labels: ndarray, shape=(n_samples,)，每个样本的簇索引 (0 ~ K-1)
        """
        # 计算距离矩阵
        distances = self._compute_distances(X)
        # 返回距离最近的簇索引
        return np.argmin(distances, axis=1)

    def transform(self, X):
        """将样本转换为到各聚类中心的距离向量

        这个变换将特征空间映射到"聚类中心距离空间"，
        可以用于后续的分类器训练（类似 RBF 网络）。

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            X_dist: ndarray, shape=(n_samples, n_clusters)
        """
        return self._compute_distances(X)

    def fit_predict(self, X):
        """训练模型并返回聚类标签（一步到位）"""
        # 先训练
        self.fit(X)
        # 再返回训练集上的聚类结果
        return self.labels_

    def __repr__(self):
        """返回模型的字符串表示"""
        return f"KMeans(n_clusters={self.n_clusters})"
