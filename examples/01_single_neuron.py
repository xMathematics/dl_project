"""
示例 1: 单层神经网络 — 从零实现逻辑回归

学习目标:
1. 理解神经元的基本结构: y = sigmoid(x @ W + b)
2. 理解二分类交叉熵损失
3. 手动实现梯度下降

本示例使用单个神经元解决 AND 逻辑门问题。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.layers.dense import Dense
from dl_from_scratch.layers.sequential import Sequential
from dl_from_scratch.activations.functions import Sigmoid
from dl_from_scratch.losses.functions import BinaryCrossEntropyLoss
from dl_from_scratch.optimizers.sgd import SGD
from dl_from_scratch.models.model import Model
from dl_from_scratch.utils.metrics import accuracy


def main():
    print("=" * 60)
    print("示例 1: 单神经元 — AND 逻辑门")
    print("=" * 60)

    # 1. 数据准备: AND 逻辑门
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=float)
    y = np.array([[0], [0], [0], [1]], dtype=float)

    print(f"\nAND 逻辑门数据:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i][0]}")

    # 2. 构建网络: 单神经元
    #    手工实现等价于: y = sigmoid(x @ W + b)
    #    (Sigmoid 由 BinaryCrossEntropyLoss 内部处理)
    network = Sequential([
        Dense(units=1),   # 线性变换: x @ W + b (输出 logits)
    ])

    # 3. 编译模型
    model = Model(network)
    model.compile(
        loss=BinaryCrossEntropyLoss(),
        optimizer=SGD(lr=1.0)  # AND 门简单，用较大学习率
    )

    # 4. 训练前看看参数
    _ = model.predict(X[:1])
    params = model.get_params()
    dense_W = [v for k, v in params.items() if 'Dense.W' in k][0]
    dense_b = [v for k, v in params.items() if 'Dense.b' in k][0]
    print(f"\n初始参数: W = {dense_W.ravel()}, b = {dense_b.ravel()}")

    # 5. 训练
    print("\n训练中...")
    history = model.fit(X, y, epochs=200, batch_size=4, verbose=True)

    # 6. 训练后参数
    params = model.get_params()
    dense_W = [v for k, v in params.items() if 'Dense.W' in k][0]
    dense_b = [v for k, v in params.items() if 'Dense.b' in k][0]
    print(f"\n训练后参数: W = {dense_W.ravel()}, b = {dense_b.ravel()}")

    # 7. 预测 (对 logits 做 sigmoid 得到概率)
    print("\n预测结果:")
    logits = model.predict(X)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred_binary = (probs > 0.5).astype(float)
    for i in range(len(X)):
        p = probs[i][0]
        pred = y_pred_binary[i][0]
        true = y[i][0]
        print(f"  {X[i]} -> 概率={p:.4f}, 预测={int(pred)}, 真实={int(true)} {'✓' if pred == true else '✗'}")

    all_correct = all((probs > 0.5).ravel() == y.ravel())
    print(f"\n全部正确: {'✓' if all_correct else '✗'}")

    # 8. 手动验证: 一个神经元等价于 sigmoid(x0*w0 + x1*w1 + b)
    print("\n手动计算验证:")
    dense_W = [v for k, v in params.items() if 'Dense.W' in k][0]
    dense_b = [v for k, v in params.items() if 'Dense.b' in k][0]
    W = dense_W.ravel()
    b = dense_b.ravel()
    for x in X:
        z = x[0] * W[0] + x[1] * W[1] + b[0]
        p = 1.0 / (1.0 + np.exp(-z))
        print(f"  z = {z:.4f}, sigmoid(z) = {p:.4f}")


if __name__ == '__main__':
    main()
