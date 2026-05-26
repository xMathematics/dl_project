"""
示例 3: 卷积神经网络 (CNN) — MNIST 手写数字识别

学习目标:
1. 理解卷积操作 (im2col 实现)
2. 理解池化层的作用
3. 理解 CNN 的层次结构: Conv -> Pool -> FC

数据集: MNIST 手写数字 (28x28 灰度图, 10 类)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.layers.convolution import Conv2D
from dl_from_scratch.layers.pooling import MaxPool2D
from dl_from_scratch.layers.flatten import Flatten
from dl_from_scratch.layers.dense import Dense
from dl_from_scratch.layers.sequential import Sequential
from dl_from_scratch.activations.functions import ReLU, Softmax
from dl_from_scratch.losses.functions import CrossEntropyLoss
from dl_from_scratch.optimizers.sgd import Adam
from dl_from_scratch.models.model import Model
from dl_from_scratch.utils.metrics import accuracy
from dl_from_scratch.utils.data import to_one_hot


def load_mnist_subset(num_train=1000, num_test=200):
    """加载 MNIST 子集 (使用 sklearn 或直接从 keras 数据集加载)"""
    try:
        from sklearn.datasets import fetch_openml
        print("从 OpenML 加载 MNIST 数据 (可能需要一些时间)...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
        X = X.values.astype(float)
        y = y.values.astype(int)

        # 归一化
        X = X / 255.0

        # 打乱并取子集
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_train = X[:num_train]
        y_train = y[:num_train]
        X_test = X[num_train:num_train + num_test]
        y_test = y[num_train:num_train + num_test]

        # 重塑为 4D (N, C, H, W)
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

        return X_train, y_train, X_test, y_test

    except ImportError:
        print("错误: 需要 sklearn 来加载 MNIST 数据")
        print("安装: pip install scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"加载 MNIST 失败: {e}")
        print("尝试生成随机数据用于演示...")
        return generate_demo_data(num_train, num_test)


def generate_demo_data(num_train, num_test):
    """生成模拟 MNIST 的随机数据用于演示"""
    np.random.seed(42)
    X_train = np.random.randn(num_train, 1, 28, 28).astype(float)
    y_train = np.random.randint(0, 10, size=num_train)
    X_test = np.random.randn(num_test, 1, 28, 28).astype(float)
    y_test = np.random.randint(0, 10, size=num_test)
    print("警告: 使用随机数据，仅用于演示代码流程")
    return X_train, y_train, X_test, y_test


def main():
    print("=" * 60)
    print("示例 3: CNN — MNIST 手写数字识别")
    print("=" * 60)

    # 1. 加载数据
    X_train, y_train, X_test, y_test = load_mnist_subset(num_train=1000, num_test=200)

    # One-hot 编码
    y_train_one_hot = to_one_hot(y_train, 10)

    print(f"\n数据形状:")
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")

    # 2. 构建 CNN
    #    结构: Conv(8, 3x3) → ReLU → MaxPool(2x2) → Conv(16, 3x3) → ReLU → MaxPool(2x2)
    #        → Flatten → Dense(64) → ReLU → Dense(10)  (输出 logits, Softmax 在 loss 内)
    network = Sequential([
        # 第一卷积块: 1->8通道, 3x3卷积
        Conv2D(out_channels=8, kernel_size=3, padding=1, stride=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        # 第二卷积块: 8->16通道, 3x3卷积
        Conv2D(out_channels=16, kernel_size=3, padding=1, stride=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),

        # 分类头 (输出 logits)
        Flatten(),
        Dense(units=64),
        ReLU(),
        Dense(units=10),
    ])

    print(f"\nCNN 结构:\n{network}")
    print(f"总参数量: {network.num_params():,}")

    # 3. 编译
    model = Model(network)
    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=0.001)
    )

    # 4. 训练
    print("\n训练 CNN... (这可能需要几分钟)")
    history = model.fit(
        X_train, y_train_one_hot,
        epochs=10,
        batch_size=32,
        verbose=True,
        X_val=X_test, y_val=y_test,
    )

    # 5. 评估 (logits -> softmax -> argmax)
    logits = model.predict(X_test)
    test_acc = accuracy(logits, y_test)
    print(f"\n测试准确率: {test_acc:.2%}")


if __name__ == '__main__':
    main()
