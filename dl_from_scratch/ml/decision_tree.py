"""决策树 — 分类 & 回归 (CART 算法)"""

import numpy as np
from collections import Counter


class _Node:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.left = left            # 左子树 (<= threshold)
        self.right = right          # 右子树 (> threshold)
        self.value = value          # 叶节点值 (分类: 多数类, 回归: 均值)


class DecisionTreeClassifier:
    """决策树分类器 (CART)

    使用基尼不纯度选择最优分裂特征和阈值。

    Args:
        max_depth: 最大深度 (None 表示不限制)
        min_samples_split: 内部节点最少样本数
        min_samples_leaf: 叶节点最少样本数
        max_features: 最大特征数 (None 表示全部)
        random_seed: 随机种子
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def _gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-15))

    def _mse(self, y):
        """均方误差 (回归用)"""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y):
        """寻找最佳分裂特征和阈值"""
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_gain = -1

        # 当前不纯度
        current_impurity = self._gini(y)

        # 选择特征子集
        features = range(n_features)
        if self.max_features is not None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            features = np.random.choice(n_features, min(self.max_features, n_features), replace=False)

        for feature in features:
            # 获取唯一阈值候选
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:
                continue

            # 取相邻值的中点作为候选阈值
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # 计算加权基尼不纯度
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                gain = current_impurity - (n_left * left_impurity + n_right * right_impurity) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or n_classes == 1:
            most_common = Counter(y).most_common(1)[0][0]
            return _Node(value=most_common)

        # 寻找最佳分裂
        feature, threshold = self._best_split(X, y)

        if feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return _Node(value=most_common)

        # 分裂
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        """训练决策树"""
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        """预测单个样本"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        """预测"""
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        """准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        depth = "∞" if self.max_depth is None else str(self.max_depth)
        return f"DecisionTreeClassifier(max_depth={depth})"


class DecisionTreeRegressor:
    """决策树回归器 (CART)

    使用 MSE 选择最优分裂。
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
        if len(y) == 0:
            return 0
        return np.var(y)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_gain = -1

        current_mse = self._mse(y)

        features = range(n_features)
        if self.max_features is not None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            features = np.random.choice(n_features, min(self.max_features, n_features), replace=False)

        for feature in features:
            thresholds = np.unique(X[:, feature])
            if len(thresholds) <= 1:
                continue
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                gain = current_mse - (n_left * left_mse + n_right * right_mse) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, _ = X.shape

        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or np.var(y) < 1e-7:
            return _Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return _Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / (ss_tot + 1e-15)

    def __repr__(self):
        depth = "∞" if self.max_depth is None else str(self.max_depth)
        return f"DecisionTreeRegressor(max_depth={depth})"
