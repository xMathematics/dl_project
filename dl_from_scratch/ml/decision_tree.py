"""决策树 — 分类 & 回归 (CART 算法)

算法原理 (CART — Classification And Regression Tree):
  1. 从根节点开始，递归地将数据二分
  2. 每次选择"最佳"分裂特征和阈值，使子节点"最纯"
  3. 到达停止条件时创建叶节点

分裂准则:
  - 分类: 基尼不纯度 (Gini Impurity) = 1 - Σ(p_k)²
  - 回归: 均方误差 (MSE) = var(y)
  - 信息增益 = 父节点不纯度 - 子节点加权不纯度

停止条件:
  - 达到最大深度 (max_depth)
  - 节点样本数小于阈值 (min_samples_split)
  - 叶节点样本数小于阈值 (min_samples_leaf)
  - 所有样本属于同一类别
"""

# Counter: 用于统计标签频率，找出多数类
from collections import Counter
import numpy as np


class _Node:
    """决策树的内部节点或叶节点

    属性:
        feature: int, 分裂使用的特征索引（叶节点为 None）
        threshold: float, 分裂阈值（叶节点为 None）
        left: _Node, 左子树（样本 <= threshold）
        right: _Node, 右子树（样本 > threshold）
        value: any, 叶节点的预测值（内部节点为 None）
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # feature: 在该节点上依据哪个特征进行分裂（列索引）
        self.feature = feature
        # threshold: 特征阈值，样本 <= threshold 去左子树，> threshold 去右子树
        self.threshold = threshold
        # left: 左子节点指针（满足条件的分支）
        self.left = left
        # right: 右子节点指针（不满足条件的分支）
        self.right = right
        # value: 叶节点的输出值，分类为多数类标签，回归为均值
        self.value = value


class DecisionTreeClassifier:
    """决策树分类器 — CART 算法

    通过递归二分构建二叉树，使用基尼不纯度选择最优分裂。

    参数:
        max_depth: int 或 None, 树的最大深度（防过拟合）
        min_samples_split: int, 内部节点所需的最小样本数
        min_samples_leaf: int, 叶节点所需的最小样本数
        max_features: int 或 None, 每次分裂考虑的最大特征数
        random_seed: int 或 None, 随机种子
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_seed=None):
        # max_depth: None=不限制，树会一直生长直到所有样本被纯化（易过拟合）
        self.max_depth = max_depth
        # min_samples_split: 节点至少有这么多样本才继续分裂
        self.min_samples_split = min_samples_split
        # min_samples_leaf: 分裂后每个子节点至少要有这么多样本
        self.min_samples_leaf = min_samples_leaf
        # max_features: 每次分裂随机选择部分特征（提高随机性，类似随机森林）
        self.max_features = max_features
        self.random_seed = random_seed
        # tree_: 训练好的树根节点，是 _Node 对象
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def _gini(self, y):
        """计算基尼不纯度 (Gini Impurity)

        公式: Gini = 1 - Σ(p_k)²
        其中 p_k = 类别 k 的样本比例

        基尼不纯度衡量数据集的不确定性:
        - 最小值 0: 所有样本属于同一类（最纯）
        - 最大值 1-1/K: 各类均匀分布（最不纯）

        参数:
            y: ndarray, 标签数组

        返回:
            gini: float, 基尼不纯度值
        """
        if len(y) == 0:
            return 0
        # np.unique(y, return_counts=True): 返回唯一值及其出现次数
        # _: 忽略唯一值列表，只保留 counts
        _, counts = np.unique(y, return_counts=True)
        # counts / len(y): 计算每个类别的概率 p_k
        probs = counts / len(y)
        # 1 - Σ(p_k)²: 基尼不纯度
        # np.sum(probs ** 2): 对概率平方求和
        return 1 - np.sum(probs ** 2)

    def _entropy(self, y):
        """计算信息熵 (Information Entropy)

        公式: H = -Σ p_k * log₂(p_k)

        信息熵与基尼类似，也是衡量不确定性的指标。
        ID3 决策树使用信息增益（熵的减少量）选择分裂。

        参数:
            y: ndarray, 标签数组

        返回:
            entropy: float, 熵值
        """
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # -Σ p_k * log₂(p_k): 信息熵
        # np.log2(): 以 2 为底的对数
        # 1e-15: 防止 log2(0) 导致 -inf
        return -np.sum(probs * np.log2(probs + 1e-15))

    def _mse(self, y):
        """均方误差 (Mean Squared Error, 回归用)

        公式: MSE = var(y) = mean((y - mean(y))²)

        参数:
            y: ndarray, 目标值数组

        返回:
            mse: float, 均方误差
        """
        if len(y) == 0:
            return 0
        # np.var(y): NumPy 的方差函数，计算均方误差
        return np.var(y)

    def _best_split(self, X, y):
        """寻找最佳分裂特征和阈值

        遍历所有特征和可能的阈值，选择使基尼增益最大的分裂方案。

        参数:
            X: ndarray, shape=(n_samples, n_features)
            y: ndarray, shape=(n_samples,)

        返回:
            best_feature: int, 最佳分裂特征索引
            best_threshold: float, 最佳分裂阈值
        """
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        # best_gain: 最大信息增益，初始为 -1（任何有效增益都应大于 -1）
        best_gain = -1

        # 当前节点的基尼不纯度（分裂前的纯度）
        current_impurity = self._gini(y)

        # ---- 特征子集选择（随机森林风格） ----
        features = range(n_features)
        if self.max_features is not None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            # np.random.choice: 从 n_features 个特征中随机选择 max_features 个
            features = np.random.choice(
                n_features, min(self.max_features, n_features), replace=False
            )

        # ---- 遍历每个候选特征 ----
        for feature in features:
            # np.unique(X[:, feature]): 获取该特征的所有唯一值
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:
                # 如果该特征只有一个取值，无法分裂
                continue

            # 取相邻唯一值的中点作为候选阈值
            # thresholds[:-1]: 除最后一个外的所有值
            # thresholds[1:]: 除第一个外的所有值
            # 两者相加除以 2 得中点
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            # ---- 遍历每个候选阈值 ----
            for threshold in thresholds:
                # 布尔掩码：特征值 <= 阈值的样本
                left_mask = X[:, feature] <= threshold
                # 取反：特征值 > 阈值的样本
                right_mask = ~left_mask

                # 检查叶节点最小样本数约束
                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                # ---- 计算基尼增益 ----
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)

                # 信息增益 = 父不纯度 - 加权子不纯度
                # 增益越大，分裂效果越好
                gain = current_impurity - (
                    n_left * left_impurity + n_right * right_impurity
                ) / n_samples

                # 更新最佳分裂
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树

        步骤:
        1. 检查是否满足停止条件 → 创建叶节点
        2. 寻找最佳分裂
        3. 无有效分裂 → 创建叶节点
        4. 分裂 → 递归构建左右子树

        参数:
            X: ndarray, 当前节点的训练数据
            y: ndarray, 当前节点的标签
            depth: int, 当前深度

        返回:
            node: _Node, 子树的根节点
        """
        n_samples = X.shape[0]
        # np.unique(y): 获取当前节点中所有不同的类别
        n_classes = len(np.unique(y))

        # ---- 停止条件检查 ----
        if ((self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or n_classes == 1):
            # 创建叶节点: 值为当前数据中最常见的类别
            # Counter(y).most_common(1)[0][0]: 取出现次数最多的标签
            most_common = Counter(y).most_common(1)[0][0]
            return _Node(value=most_common)

        # ---- 寻找最佳分裂 ----
        feature, threshold = self._best_split(X, y)

        # 找不到有效分裂（所有特征增益均为负）→ 创建叶节点
        if feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return _Node(value=most_common)

        # ---- 分裂并递归构建 ----
        # 左子树: 特征值 <= 阈值的样本
        left_mask = X[:, feature] <= threshold
        # 右子树: 特征值 > 阈值的样本
        right_mask = ~left_mask

        # 递归构建左右子树
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # 返回内部节点
        return _Node(
            feature=feature,
            threshold=threshold,
            left=left_node,
            right=right_node
        )

    def fit(self, X, y):
        """训练决策树模型

        参数:
            X: ndarray, shape=(n_samples, n_features)
            y: ndarray, shape=(n_samples,)

        返回:
            self: 训练后的模型
        """
        # 记录类别数和特征数
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        # 递归构建整棵树
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        """递归预测单个样本

        从根节点开始，根据特征值与阈值比较，
        决定向左还是向右，直到到达叶节点。

        参数:
            x: ndarray, shape=(n_features,)，单个样本
            node: _Node, 当前节点

        返回:
            value: 预测的类别标签
        """
        # 到达叶节点（有 value 属性）→ 返回预测值
        if node.value is not None:
            return node.value
        # 比较特征值和阈值，决定去左子树还是右子树
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        """批量预测

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            predictions: ndarray, shape=(n_samples,)
        """
        # 对每个样本调用 _predict_one
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        """计算分类准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        depth = "∞" if self.max_depth is None else str(self.max_depth)
        return f"DecisionTreeClassifier(max_depth={depth})"


class DecisionTreeRegressor:
    """决策树回归器 — CART 算法

    与分类树基本相同，区别:
    - 分裂准则使用 MSE 而非基尼不纯度
    - 叶节点输出为均值而非多数类

    参数:
        max_depth: int 或 None, 最大深度
        min_samples_split: int, 内部节点最少样本数
        min_samples_leaf: int, 叶节点最少样本数
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.tree_ = None

    def _mse(self, y):
        """均方误差: MSE = var(y)"""
        if len(y) == 0:
            return 0
        # np.var(y): 方差 = mean((y - mean(y))²)
        return np.var(y)

    def _best_split(self, X, y):
        """寻找回归的最佳分裂（使用 MSE 增益）"""
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_gain = -1

        # 当前节点的 MSE
        current_mse = self._mse(y)

        # 选择特征子集
        features = range(n_features)
        if self.max_features is not None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            features = np.random.choice(
                n_features, min(self.max_features, n_features), replace=False
            )

        for feature in features:
            # 获取候选阈值（唯一值的中点）
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:
                continue
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                # 计算 MSE 增益
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                gain = current_mse - (
                    n_left * left_mse + n_right * right_mse
                ) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建回归树

        参数:
            X: ndarray, 当前节点的训练数据
            y: ndarray, 当前节点的目标值
            depth: int, 当前深度

        返回:
            node: _Node, 子树根节点
        """
        n_samples = X.shape[0]

        # 停止条件: 达到最大深度 / 样本太少 / 目标值方差太小（几乎无变化）
        if ((self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or np.var(y) < 1e-7):
            # np.mean(y): 叶节点输出为目标值的均值
            return _Node(value=np.mean(y))

        # 寻找最佳分裂
        feature, threshold = self._best_split(X, y)

        # 无有效分裂 → 创建叶节点
        if feature is None:
            return _Node(value=np.mean(y))

        # 分裂并递归
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature=feature, threshold=threshold,
            left=left_node, right=right_node
        )

    def fit(self, X, y):
        """训练回归树"""
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        """递归预测单个样本的目标值"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        """计算 R² 决定系数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        depth = "∞" if self.max_depth is None else str(self.max_depth)
        return f"DecisionTreeRegressor(max_depth={depth})"
