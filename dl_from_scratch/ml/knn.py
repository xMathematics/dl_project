"""K 近邻 (K-Nearest Neighbors) — 分类 & 回归

算法原理:
  KNN 是一种"惰性学习"(Lazy Learning) 算法，没有显式的训练过程。
  对新样本:
  1. 计算它与所有训练样本的距离
  2. 选出最近的 K 个邻居
  3. 分类: 多数投票 | 回归: 取均值

距离度量:
  - 欧氏距离: sqrt(Σ(x-y)²)
  - 曼哈顿距离: Σ|x-y|
  - 余弦距离: 1 - cos(x,y)
"""

# Counter: Python 标准库的计数器，用于统计列表中元素出现的频率
from collections import Counter
import numpy as np


class KNN:
    """K 近邻模型

    惰性学习器，训练时仅保存数据，预测时实时计算距离。

    参数:
        k: int, 邻居数量（奇数有助于避免平局）
        distance_metric: str, 距离度量方式
        task: str, 'classification' 或 'regression'
    """

    def __init__(self, k=5, distance_metric='euclidean', task='classification'):
        # k: 最近邻的数量。k 越小决策边界越复杂（易过拟合），越大越平滑（易欠拟合）
        self.k = k
        # distance_metric: 距离计算方式，影响"近邻"的定义
        self.distance_metric = distance_metric
        # task: 任务类型，决定预测时用投票还是平均
        self.task = task
        # X_train: 训练特征矩阵，KNN 需要保存所有训练数据
        self.X_train = None
        # y_train: 训练标签
        self.y_train = None

    def _distance(self, x1, x2):
        """计算两个样本向量之间的距离

        支持三种距离度量:
        1. 欧氏距离 (euclidean): 最常用的直线距离
        2. 曼哈顿距离 (manhattan): 坐标轴绝对距离之和
        3. 余弦距离 (cosine): 衡量方向差异（常用于文本）

        参数:
            x1, x2: 形状相同的向量，shape=(n_features,)

        返回:
            dist: float, 距离值（非负）
        """
        if self.distance_metric == 'euclidean':
            # 欧氏距离: d = sqrt(Σ(x1_i - x2_i)²)
            # x1 - x2: 逐元素相减 (NumPy 广播)
            # ** 2: 对每个元素平方
            # np.sum(): 对所有元素求和
            # np.sqrt(): 开平方根
            return np.sqrt(np.sum((x1 - x2) ** 2))

        elif self.distance_metric == 'manhattan':
            # 曼哈顿距离: d = Σ|x1_i - x2_i|
            # np.abs(): 计算每个元素的绝对值
            # np.sum(): 求和
            return np.sum(np.abs(x1 - x2))

        elif self.distance_metric == 'cosine':
            # 余弦距离: d = 1 - cos(θ) = 1 - (x·y) / (||x||·||y||)
            # np.dot(x1, x2): 向量点积，即 x1 · x2
            dot = np.dot(x1, x2)
            # np.linalg.norm(x): 计算 L2 范数（模长），即 sqrt(Σx_i²)
            # 1e-15: 防止零向量导致除零
            norm = np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
            # 余弦距离 = 1 - 余弦相似度
            # 余弦相似度 = 1（方向相同），= 0（正交），= -1（方向相反）
            return 1 - dot / norm
        else:
            raise ValueError(f"未知距离度量: {self.distance_metric}")

    def fit(self, X, y):
        """训练 KNN 模型

        KNN 的"训练"就是记住所有训练数据，没有任何计算。
        这也是为什么它被称为"惰性学习"。

        参数:
            X: ndarray, shape=(n_samples, n_features)，训练特征
            y: ndarray, shape=(n_samples,)，训练标签

        返回:
            self: 模型实例
        """
        # 保存训练数据到实例变量，供 predict 时使用
        self.X_train = X
        self.y_train = y
        return self

    def _predict_one(self, x):
        """预测单个样本的标签

        步骤:
        1. 计算到所有训练样本的距离
        2. 找出最近的 K 个样本
        3. 分类: 投票选多数类 | 回归: 取均值

        参数:
            x: ndarray, shape=(n_features,)，单个待测样本

        返回:
            label: 预测的类别或数值
        """
        # 列表推导式: 对每个训练样本 x_train 计算距离
        # 结果 distances 是一个长度为 n_samples 的列表
        distances = [self._distance(x, x_train) for x_train in self.X_train]

        # np.argsort(distances): 返回距离从小到大的索引数组
        # [:self.k]: 取前 K 个（最近的 K 个邻居）的索引
        k_indices = np.argsort(distances)[:self.k]

        # 用索引从训练标签中取出 K 个邻居的标签
        k_labels = self.y_train[k_indices]

        if self.task == 'classification':
            # ---- 分类: 多数投票 ----
            # Counter(k_labels): 统计每个标签出现的次数
            #   .most_common(1): 返回出现次数最多的 (标签, 次数) 列表
            #   [0][0]: 取出次数最多的标签值
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        else:
            # ---- 回归: 取均值 ----
            # np.mean(k_labels): 对 K 个邻居的标签值求平均
            return np.mean(k_labels)

    def predict(self, X):
        """批量预测多个样本

        对每个样本调用 _predict_one 进行预测。

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            predictions: ndarray, shape=(n_samples,)，预测结果
        """
        # 列表推导式: 对 X 的每一行（每个样本）调用 _predict_one
        # np.array(): 将列表转换为 NumPy 数组
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        """计算模型得分

        分类: 准确率 (Accuracy) = 预测正确的比例
        回归: R² 决定系数 (Coefficient of Determination)

        参数:
            X: ndarray, 特征矩阵
            y: ndarray, 真实标签

        返回:
            score: float, 得分值
        """
        y_pred = self.predict(X)
        if self.task == 'classification':
            # 准确率: 预测正确的样本比例
            # y_pred == y: 逐元素比较，返回布尔数组
            # np.mean(布尔数组): True=1, False=0 求平均得准确率
            return np.mean(y_pred == y)
        else:
            # R² = 1 - SS_res / SS_tot
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        """返回模型的字符串表示"""
        return f"KNN(k={self.k}, metric={self.distance_metric})"
