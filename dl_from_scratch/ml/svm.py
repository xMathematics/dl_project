"""支持向量机 (SVM) — 使用 SMO 算法"""

import numpy as np


class SVM:
    """支持向量机 (二分类)

    使用简化的 SMO (Sequential Minimal Optimization) 算法训练。

    Kernel 选项:
    - 'linear': K(x, y) = x · y
    - 'poly':   K(x, y) = (γ x · y + r)^d
    - 'rbf':    K(x, y) = exp(-γ ||x - y||²)

    Args:
        C: 正则化参数 (软间隔)
        kernel: 核函数类型
        degree: 多项式核的度数
        gamma: RBF/多项式核参数 ('scale' 或 float)
        coef0: 多项式核的常数项
        max_iter: SMO 最大迭代次数
        tol: 容差
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
        self.alpha = None       # 拉格朗日乘子
        self.support_vectors_ = None
        self.support_labels_ = None
        self.b = 0.0
        self._gamma = None

    def _kernel_func(self, x1, x2):
        """计算核函数"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self._gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self._gamma * np.dot(diff, diff))
        else:
            raise ValueError(f"未知核函数: {self.kernel}")

    def _kernel_matrix(self, X):
        """计算核矩阵 K_ij = K(X_i, X_j)"""
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel_func(X[i], X[j])
        return K

    def fit(self, X, y):
        """训练 SVM — 简化的 SMO 算法

        Args:
            X: shape=(n_samples, n_features)
            y: shape=(n_samples,), 取值 {-1, +1}
        """
        n_samples, n_features = X.shape
        y = y.astype(float)

        # 设置 gamma
        if self.gamma == 'scale':
            self._gamma = 1.0 / (n_features * X.var())
        else:
            self._gamma = self.gamma

        # 初始化
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # 预计算核矩阵
        K = self._kernel_matrix(X)

        # SMO 迭代
        for it in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            num_changed = 0

            for i in range(n_samples):
                # 选择 j ≠ i
                j = np.random.randint(n_samples)
                while j == i:
                    j = np.random.randint(n_samples)

                # 计算误差
                E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                E_j = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]

                # 保存旧值
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                # 计算边界
                if y[i] != y[j]:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])

                if abs(L - H) < 1e-10:
                    continue

                # 计算 η
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # 更新 α_j
                self.alpha[j] -= y[j] * (E_i - E_j) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    continue

                # 更新 α_i
                self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                # 更新 b
                b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                     - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                     - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                num_changed += 1

            # 检查收敛
            if num_changed == 0:
                break

        # 提取支持向量
        sv_mask = self.alpha > 1e-5
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]
        self.alpha = self.alpha[sv_mask]

        return self

    def decision_function(self, X):
        """计算决策函数值 f(x) = Σ α_i y_i K(x_i, x) + b"""
        n_samples = X.shape[0]
        decision = np.zeros(n_samples)

        for i in range(n_samples):
            for sv, alpha, sv_y in zip(self.support_vectors_, self.alpha, self.support_labels_):
                decision[i] += alpha * sv_y * self._kernel_func(X[i], sv)
            decision[i] += self.b

        return decision

    def predict(self, X):
        """预测类别 {-1, +1}"""
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        """准确率"""
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"SVM(C={self.C}, kernel={self.kernel})"
