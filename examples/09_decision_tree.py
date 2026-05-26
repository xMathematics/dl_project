"""
示例 9: 决策树 — 从零实现

学习目标:
1. 理解决策树的分裂准则 (基尼不纯度)
2. 理解递归建树过程
3. 观察过拟合与剪枝
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.ml.decision_tree import DecisionTreeClassifier
from dl_from_scratch.utils.data import make_circles, train_test_split


def main():
    print("=" * 60)
    print("示例 9: 决策树 — 同心圆分类")
    print("=" * 60)

    # 1. 生成非线性数据
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, seed=42)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25, seed=42)

    print(f"\n训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 2. 不同深度的对比
    print("\n不同深度对准确率的影响:")
    for depth in [1, 2, 3, 5, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_seed=42)
        dt.fit(X_train, y_train)
        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)

        depth_str = "∞" if depth is None else str(depth)
        print(f"  深度={depth_str:3s}: 训练准确率={train_acc:.2%}, "
              f"测试准确率={test_acc:.2%}")

    # 3. 分析
    print("\n" + "=" * 60)
    print("决策树核心要点:")
    print("  - 分裂准则: 基尼不纯度 / 信息增益 / MSE")
    print("  - 深度过深 → 过拟合 (训练 100%, 测试下降)")
    print("  - 深度过浅 → 欠拟合 (训练和测试都低)")
    print("  - 可以通过剪枝、限制深度、最小叶节点数来防止过拟合")


if __name__ == '__main__':
    main()
