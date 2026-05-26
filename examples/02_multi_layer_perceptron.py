"""
示例 2: 多层感知机 (MLP) — 从零实现反向传播

学习目标:
1. 理解多层网络的前向传播
2. 理解链式法则 (反向传播的本质)
3. 感受隐藏层带来的非线性表达能力

本示例使用 MLP 分类半月形数据集 (非线性可分)。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.layers.dense import Dense
from dl_from_scratch.layers.sequential import Sequential
from dl_from_scratch.activations.functions import Tanh
from dl_from_scratch.losses.functions import BinaryCrossEntropyLoss
from dl_from_scratch.optimizers.sgd import SGD
from dl_from_scratch.models.model import Model
from dl_from_scratch.utils.data import make_moons, train_test_split


def plot_decision_boundary(model, X, y, title="决策边界"):
    """绘制决策边界 (需要 matplotlib)"""
    try:
        import matplotlib.pyplot as plt

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
    except ImportError:
        print("(matplotlib 未安装，跳过绘图)")


def main():
    print("=" * 60)
    print("示例 2: 多层感知机 (MLP) — 半月形分类")
    print("=" * 60)

    # 1. 生成数据
    X, y = make_moons(n_samples=200, noise=0.15, seed=42)
    y = y.reshape(-1, 1)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    print(f"\n数据: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    print(f"非线性可分数据集 — 单层网络无法解决!")

    # 2. 构建 MLP
    #    输入层(2) -> 隐藏层(16, Tanh) -> 输出层(1, logits)
    #    (Sigmoid 由 BinaryCrossEntropyLoss 内部处理)
    network = Sequential([
        Dense(units=16),
        Tanh(),
        Dense(units=1),  # 输出 logits
    ])

    print(f"\n网络结构:\n{network}")

    # 3. 编译
    model = Model(network)
    model.compile(
        loss=BinaryCrossEntropyLoss(),
        optimizer=SGD(lr=0.5)
    )

    # 4. 训练
    print("\n训练中...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        verbose=True,
        X_val=X_test, y_val=y_test,
    )

    # 5. 评估 (logits -> sigmoid -> 概率 -> 二值预测)
    logits_train = model.predict(X_train)
    probs_train = 1.0 / (1.0 + np.exp(-logits_train))
    y_pred_train = (probs_train > 0.5).astype(int).ravel()

    logits_test = model.predict(X_test)
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))
    y_pred_test = (probs_test > 0.5).astype(int).ravel()

    train_acc = np.mean(y_pred_train == y_train.ravel())
    test_acc = np.mean(y_pred_test == y_test.ravel())

    print(f"\n训练准确率: {train_acc:.2%}")
    print(f"测试准确率: {test_acc:.2%}")

    # 6. 尝试对比: 单层网络效果
    print("\n对比: 单层网络 (Dense -> Sigmoid in loss) 的效果:")
    simple_network = Sequential([
        Dense(units=1),
    ])
    simple_model = Model(simple_network)
    simple_model.compile(
        loss=BinaryCrossEntropyLoss(),
        optimizer=SGD(lr=0.5)
    )
    simple_model.fit(X_train, y_train, epochs=500, batch_size=16, verbose=False)
    simple_logits = simple_model.predict(X_test)
    simple_probs = 1.0 / (1.0 + np.exp(-simple_logits))
    simple_pred = (simple_probs > 0.5).astype(int).ravel()
    simple_acc = np.mean(simple_pred == y_test.ravel())
    print(f"  单层网络测试准确率: {simple_acc:.2%} (非线性可分数据无法解决!)")

    # 7. 可视化
    plot_decision_boundary(model, X, y, "MLP 决策边界")


if __name__ == '__main__':
    main()
