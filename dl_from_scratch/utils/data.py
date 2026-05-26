"""数据处理工具"""

import numpy as np


def make_moons(n_samples=100, noise=0.1, seed=None):
    """生成半月形数据集 (二分类)"""
    if seed is not None:
        np.random.seed(seed)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # 外圈
    outer_linspace = np.linspace(0, np.pi, n_samples_out)
    outer_x = np.cos(outer_linspace)
    outer_y = np.sin(outer_linspace)
    outer = np.column_stack([outer_x, outer_y])

    # 内圈
    inner_linspace = np.linspace(0, np.pi, n_samples_in)
    inner_x = 1 - np.cos(inner_linspace)
    inner_y = 1 - np.sin(inner_linspace) - 0.5
    inner = np.column_stack([inner_x, inner_y])

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    # 添加噪声
    X += np.random.randn(*X.shape) * noise

    return X, y


def make_circles(n_samples=100, noise=0.05, factor=0.5, seed=None):
    """生成同心圆数据集"""
    if seed is not None:
        np.random.seed(seed)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # 外圈
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out)
    outer = np.column_stack([
        np.cos(linspace_out), np.sin(linspace_out)
    ])

    # 内圈
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in)
    inner = np.column_stack([
        np.cos(linspace_in) * factor, np.sin(linspace_in) * factor
    ])

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    X += np.random.randn(*X.shape) * noise
    return X, y


def to_one_hot(y, num_classes=None):
    """将整数标签转换为 one-hot 编码

    Args:
        y: 整数标签, shape=(n_samples,)
        num_classes: 类别数

    Returns:
        one-hot 编码, shape=(n_samples, num_classes)
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def batch_iterator(X, y=None, batch_size=32, shuffle=True):
    """批次迭代器

    Args:
        X: 数据, shape=(n_samples, ...)
        y: 标签 (可选)
        batch_size: 批次大小
        shuffle: 是否打乱

    Yields:
        (X_batch, y_batch) 或 X_batch
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]

        if y is not None:
            yield X[batch_indices], y[batch_indices]
        else:
            yield X[batch_indices]


def train_test_split(X, y, test_size=0.2, seed=None):
    """划分训练集和测试集"""
    if seed is not None:
        np.random.seed(seed)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (X[train_indices], y[train_indices],
            X[test_indices], y[test_indices])


def normalize(X, mean=None, std=None):
    """标准化: (X - mean) / std"""
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0) + 1e-8

    return (X - mean) / std, mean, std


def standardize(X, min_val=None, max_val=None):
    """归一化到 [0, 1]: (X - min) / (max - min)"""
    if min_val is None:
        min_val = np.min(X, axis=0)
    if max_val is None:
        max_val = np.max(X, axis=0)

    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1.0, range_val)

    return (X - min_val) / range_val, min_val, max_val
