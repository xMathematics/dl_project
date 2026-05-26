"""K 近邻 (K-Nearest Neighbors) — 分类 & 回归"""

import numpy as np
from collections import Counter


class KNN:
    """K 近邻算法

    原理: 对于新样本，找到训练集中 K 个最近邻，通过投票/平均做预测。

    Args:
        k: 邻居数量
        distance_metric: 距离度量 ('euclidean', 'manhattan', 'cosine')
        task: 'classification' 或 'regression'
    """

    def __init__(self, k=5, distance_metric='euclidean', task='classification'):
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.X_train = None
        self.y_train = None

    def _distance(self, x1, x2):
        """计算两个样本之间的距离"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            dot = np.dot(x1, x2)
            norm = np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
            return 1 - dot / norm
        else:
            raise ValueError(f"未知距离度量: {self.distance_metric}")

    def fit(self, X, y):
        """训练 — KNN 只是记住数据"""
        self.X_train = X
        self.y_train = y
        return self

    def _predict_one(self, x):
        """预测单个样本"""
        # 计算到所有训练样本的距离
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # 获取 K 个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        if self.task == 'classification':
            # 投票
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        else:
            # 平均
            return np.mean(k_labels)

    def predict(self, X):
        """预测多个样本"""
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        """准确率 (分类) 或 R² (回归)"""
        y_pred = self.predict(X)
        if self.task == 'classification':
            return np.mean(y_pred == y)
        else:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        return f"KNN(k={self.k}, metric={self.distance_metric})"
