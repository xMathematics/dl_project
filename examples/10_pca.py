"""
示例 10: PCA 降维 — 从零实现

学习目标:
1. 理解协方差矩阵和特征值分解
2. 理解方差解释比例
3. 将高维数据降到 2D 进行可视化
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.ml.pca import PCA


def main():
    print("=" * 60)
    print("示例 10: PCA 降维可视化")
    print("=" * 60)

    # 1. 生成高维数据 (实际只有少数几个方向有变化)
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # 在 3 个主方向上生成数据，其余维度是噪声
    t = np.random.uniform(-2, 2, n_samples)
    X = np.column_stack([
        t * 2 + np.random.randn(n_samples) * 0.3,     # 主方向 1
        t * 1.5 + np.random.randn(n_samples) * 0.3,    # 主方向 2
        np.sin(t) + np.random.randn(n_samples) * 0.2,   # 主方向 3
    ])
    # 添加随机噪声维度
    X = np.c_[X, np.random.randn(n_samples, n_features - 3) * 0.5]

    print(f"\n数据: {X.shape[0]} 样本, {X.shape[1]} 维 (实际只有 3 个主方向)")

    # 2. PCA 降维
    pca = PCA(n_components=5)
    pca.fit(X)

    print("\n各主成分的方差解释比例:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        cumulative = np.sum(pca.explained_variance_ratio_[:i+1])
        print(f"  主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%) "
              f"[累计: {cumulative*100:.2f}%]")

    # 3. 降维到 2D
    print("\n降维到 2D...")
    X_2d = pca.transform(X)
    print(f"  降维后形状: {X_2d.shape}")

    # 4. 重建
    print("\n重建误差 (降维到 5D 再恢复):")
    pca5 = PCA(n_components=5)
    X_pca = pca5.fit_transform(X)
    X_reconst = pca5.inverse_transform(X_pca)
    mse = np.mean((X - X_reconst) ** 2)
    print(f"  MSE = {mse:.6f}")

    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X)
    X_reconst2 = pca2.inverse_transform(X_pca2)
    mse2 = np.mean((X - X_reconst2) ** 2)
    print(f"  保留 2 个主成分的 MSE = {mse2:.6f} (信息损失更大)")

    # 5. 分析
    print("\n" + "=" * 60)
    print("PCA 核心要点:")
    print("  - 通过特征值分解找到方差最大的方向")
    print("  - 方差解释比例 = 特征值 / 总特征值之和")
    print("  - 通常保留累计方差 > 95% 的主成分")
    print("  - PCA 是线性降维方法，对非线性数据效果有限")


if __name__ == '__main__':
    main()
