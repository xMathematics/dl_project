"""
示例 8: K 均值聚类 (K-Means) — 从零实现

学习目标:
1. 理解无监督学习的聚类思想
2. 理解 K-Means 算法流程 (分配 → 更新 → 收敛)
3. 理解 K-Means++ 初始化
4. 学习如何选择 K (肘部法则)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.ml.kmeans import KMeans


def generate_cluster_data(n_samples=300, n_clusters=4, seed=42):
    """生成聚类数据"""
    np.random.seed(seed)
    centers = np.random.randn(n_clusters, 2) * 3
    X = []
    for i in range(n_clusters):
        n = n_samples // n_clusters
        cluster = centers[i] + np.random.randn(n, 2) * 0.6
        X.append(cluster)
    X = np.vstack(X)
    true_labels = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
    return X, true_labels, centers


def main():
    print("=" * 60)
    print("示例 8: K-Means 聚类")
    print("=" * 60)

    # 1. 生成数据
    X, true_labels, true_centers = generate_cluster_data(n_samples=300, n_clusters=4)
    print(f"\n数据: {X.shape[0]} 样本, 4 个真实聚类")

    # 2. 训练 K-Means
    print("\n训练 K-Means (K-Means++ 初始化)...")
    kmeans = KMeans(n_clusters=4, random_seed=42)
    kmeans.fit(X)

    print(f"  迭代后聚类中心:\n  {kmeans.centroids}")
    print(f"  惯量 (inertia): {kmeans.inertia_:.4f}")
    print(f"  各聚类样本数: {np.bincount(kmeans.labels_)}")

    # 3. 评估
    from dl_from_scratch.utils.metrics import accuracy
    # 调整标签 (K-Means 标签顺序可能与真实标签不同)
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    try:
        from sklearn.utils.linear_assignment_ import linear_assignment
    except ImportError:
        # 简单匹配
        def match_labels(true, pred, k):
            from collections import Counter
            mapping = {}
            for i in range(k):
                mask = pred == i
                if mask.sum() > 0:
                    mapping[i] = Counter(true[mask]).most_common(1)[0][0]
            return np.array([mapping.get(p, -1) for p in pred])

    matched = match_labels(true_labels, kmeans.labels_, 4)
    acc = np.mean(matched == true_labels)
    print(f"  聚类准确率: {acc:.2%}")

    # 4. 肘部法则 — 选择 K
    print("\n肘部法则 (选择最优 K):")
    inertias = []
    for k in range(1, 9):
        km = KMeans(n_clusters=k, random_seed=42)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  K={k}: inertia={km.inertia_:.2f}")

    # 寻找"肘部"
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow = np.argmax(diffs2) + 2  # +2 because of double diff
    print(f"\n推荐 K={elbow} (肘部位置)")


if __name__ == '__main__':
    main()
