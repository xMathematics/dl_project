"""
示例 7: K 近邻 (KNN) — 从零实现

学习目标:
1. 理解惰性学习 (Lazy Learning) 的概念
2. 理解不同距离度量对结果的影响
3. 观察 K 值对决策边界的影响
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.ml.knn import KNN
from dl_from_scratch.utils.data import make_moons, train_test_split


def main():
    print("=" * 60)
    print("示例 7: K 近邻 (KNN) — 半月形分类")
    print("=" * 60)

    # 1. 生成数据
    X, y = make_moons(n_samples=150, noise=0.2, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, seed=42)

    print(f"\n训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 2. 不同 K 值的对比
    print("\n不同 K 值的准确率:")
    for k in [1, 3, 5, 11, 21]:
        knn = KNN(k=k, distance_metric='euclidean', task='classification')
        knn.fit(X_train, y_train)
        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)
        print(f"  K={k:2d}: 训练准确率={train_acc:.2%}, 测试准确率={test_acc:.2%}")

    # 3. 不同距离度量的对比
    print("\n不同距离度量 (K=5):")
    for metric in ['euclidean', 'manhattan', 'cosine']:
        knn = KNN(k=5, distance_metric=metric, task='classification')
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        print(f"  {metric:10s}: 测试准确率={acc:.2%}")

    # 4. 分析
    print("\n" + "=" * 60)
    print("KNN 本质:")
    print("  - 没有显式训练过程 → 惰性学习")
    print("  - K 越小 → 决策边界越复杂 → 容易过拟合")
    print("  - K 越大 → 决策边界越平滑 → 容易欠拟合")
    print("  - 距离度量决定了"近邻"的定义")


if __name__ == '__main__':
    main()
