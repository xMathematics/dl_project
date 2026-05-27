"""支持向量机 (SVM) — 使用 SMO 算法

数学原理:
  SVM 寻找一个超平面 w·x + b = 0，使两类样本的间隔最大化。

  优化问题 (软间隔):
    min ½||w||² + C·Σξ_i
    s.t. y_i(w·x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0

  对偶问题 (引入核函数):
    max Σα_i - ½ΣΣα_iα_jy_i y_j K(x_i, x_j)
    s.t. 0 ≤ α_i ≤ C, Σα_i y_i = 0

  SMO (序列最小优化) 算法迭代求解 α。

  核函数:
  - linear: K(x,y) = x·y
  - poly:   K(x,y) = (γx·y + r)^d
  - rbf:    K(x,y) = exp(-γ||x-y||²)
"""

import numpy as np


class SVM:
    """支持向量机 (二分类)

    使用 SMO (Sequential Minimal Optimization) 算法求解对偶问题。
    支持线性核、多项式核和 RBF 核。

    参数:
        C: float, 正则化参数。越小 → 更宽的间隔 + 更多容错
        kernel: str, 核函数类型: 'linear', 'poly', 'rbf'
        degree: int, 多项式核的度数 (仅 poly)
        gamma: str 或 float, RBF/多项式核参数
        coef0: float, 多项式核的常数项 (仅 poly)
        max_iter: int, SMO 最大迭代次数
        tol: float, 收敛容差
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, max_iter=1000, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        # alpha: 拉格朗日乘子，仅支持向量对应的 α > 0
        self.alpha = None
        # support_vectors_: 支持向量（α > 0 的样本）
        self.support_vectors_ = None
        # support_labels_: 支持向量的标签
        self.support_labels_ = None
        # b: 偏置项
        self.b = 0.0
        # _gamma: 实际的 gamma 值
        self._gamma = None

    def _kernel_func(self, x1, x2):
        """计算两个样本之间的核函数值 K(x1, x2)

        核函数将样本隐式映射到高维空间，使线性不可分的数据变得可分。
        "核技巧": 不需要显式计算高维映射，直接计算内积。

        参数:
            x1, x2: ndarray, shape=(n_features,)，两个样本

        返回:
            value: float, 核函数值
        """
        if self.kernel == 'linear':
            # 线性核: K(x,y) = x · y
            # np.dot(x1, x2): 向量点积
            return np.dot(x1, x2)

        elif self.kernel == 'poly':
            # 多项式核: K(x,y) = (γ x·y + r)^d
            # 可以捕捉特征之间的交互
            return (self._gamma * np.dot(x1, x2) + self.coef0) ** self.degree

        elif self.kernel == 'rbf':
            # RBF 核 (高斯核): K(x,y) = exp(-γ||x-y||²)
            # 最常用的核函数，可以映射到无穷维空间
            diff = x1 - x2
            # np.dot(diff, diff): 计算 ||x-y||²（差的平方范数）
            # np.exp: e^(-γ·dist²)，值域 (0,1]，样本越近值越大
            return np.exp(-self._gamma * np.dot(diff, diff))
        else:
            raise ValueError(f"未知核函数: {self.kernel}")

    def _kernel_matrix(self, X):
        """计算整个数据集的核矩阵 (Gram 矩阵)

        K[i, j] = K(x_i, x_j), shape=(n_samples, n_samples)

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            K: ndarray, shape=(n_samples, n_samples)，对称正定矩阵
        """
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # 计算每对样本的核函数值
                K[i, j] = self._kernel_func(X[i], X[j])
        return K

    def fit(self, X, y):
        """训练 SVM 模型 — 使用简化的 SMO 算法

        SMO 每次选择两个 α_i, α_j 进行优化，固定其他 α。
        通过迭代所有 α 对，逐步收敛到全局最优。

        参数:
            X: ndarray, shape=(n_samples, n_features)，训练数据
            y: ndarray, shape=(n_samples,)，标签，取值 {-1, +1}

        返回:
            self: 训练后的模型
        """
        n_samples, n_features = X.shape
        # 确保 y 是浮点型 {-1, +1}
        y = y.astype(float)

        # ---- 设置 gamma ----
        if self.gamma == 'scale':
            # 'scale': 根据特征数自适应，gamma = 1/(n_features * X.var())
            # X.var(): 所有数据的方差
            self._gamma = 1.0 / (n_features * X.var())
        else:
            self._gamma = self.gamma

        # ---- 初始化 ----
        # 所有 α 初始为 0
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # 预计算核矩阵 K (避免重复计算)
        K = self._kernel_matrix(X)

        # ---- SMO 迭代 ----
        for it in range(self.max_iter):
            # alpha_prev: 记录迭代前的 α，用于检查收敛
            alpha_prev = self.alpha.copy()
            # num_changed: 本轮有多少 α 对发生了变化
            num_changed = 0

            # 遍历所有样本，对每个 i 随机选一个 j ≠ i
            for i in range(n_samples):
                # ---- 选择第二个 α ----
                # np.random.randint(n_samples): 随机选择一个 j
                j = np.random.randint(n_samples)
                while j == i:
                    j = np.random.randint(n_samples)

                # ---- 计算误差 E = f(x) - y ----
                # E_i = Σα_k y_k K(x_k, x_i) + b - y_i
                # np.sum(self.alpha * y * K[:, i]): 对 k 求和
                E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                E_j = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]

                # 保存旧 α 值
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                # ---- 计算 α_j 的边界 [L, H] ----
                if y[i] != y[j]:
                    # 不同类: L = max(0, α_j - α_i), H = min(C, C + α_j - α_i)
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    # 同类: L = max(0, α_i + α_j - C), H = min(C, α_i + α_j)
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])

                # 如果边界非常接近，跳过这对 α
                if abs(L - H) < 1e-10:
                    continue

                # ---- 计算 η = K(ii) + K(jj) - 2K(ij) ----
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                # η 必须为负（保证目标函数是凸的）
                if eta >= 0:
                    continue

                # ---- 更新 α_j ----
                # 裁剪前的更新量: α_j_new = α_j_old - y_j(E_i - E_j) / η
                self.alpha[j] -= y[j] * (E_i - E_j) / eta
                # np.clip(α_j_new, L, H): 将 α_j 限制在 [L, H] 区间
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # 如果 α_j 变化太小，跳过
                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    continue

                # ---- 更新 α_i ----
                # α_i_new = α_i_old + y_i*y_j*(α_j_old - α_j_new)
                self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                # ---- 更新偏置 b ----
                b1 = (self.b - E_i
                      - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i]
                      - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j])
                b2 = (self.b - E_j
                      - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j]
                      - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j])

                # 根据 α 是否在边界内选择 b
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                num_changed += 1

            # ---- 收敛检查 ----
            # 如果所有 α 对都没有变化，认为已收敛
            if num_changed == 0:
                break

        # ---- 提取支持向量 ----
        # 支持向量: α > 1e-5 的样本
        # sv_mask: 布尔掩码，True 表示该样本是支持向量
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]
        # 只保留支持向量的 α
        self.alpha = self.alpha[sv_mask]

        return self

    def decision_function(self, X):
        """计算决策函数值 f(x)

        f(x) = Σ α_i y_i K(x_i, x) + b
        只对支持向量求和（非支持向量的 α_i = 0）。

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            decision: ndarray, shape=(n_samples,)，决策值
                f(x) > 0 → 正类 (+1)
                f(x) < 0 → 负类 (-1)
        """
        n_samples = X.shape[0]
        decision = np.zeros(n_samples)

        # 对每个待测样本
        for i in range(n_samples):
            # 只对支持向量计算核函数（其他样本 α=0，不起作用）
            for sv, alpha, sv_y in zip(
                self.support_vectors_, self.alpha, self.support_labels_
            ):
                # Σ α_i y_i K(x_i, x)
                decision[i] += alpha * sv_y * self._kernel_func(X[i], sv)
            # + b
            decision[i] += self.b

        return decision

    def predict(self, X):
        """预测类别标签

        参数:
            X: ndarray, shape=(n_samples, n_features)

        返回:
            y_pred: ndarray, shape=(n_samples,), 取值 {-1, +1}
        """
        # np.sign(f(x)): 决策函数值的符号
        #   > 0 → +1, < 0 → -1, = 0 → 0
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        """计算分类准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"SVM(C={self.C}, kernel={self.kernel})"
