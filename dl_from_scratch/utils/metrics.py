"""评估指标"""

import numpy as np


def accuracy(y_pred, y_true):
    """计算分类准确率

    Args:
        y_pred: 模型预测 (概率或 logits), shape=(n_samples, num_classes)
                或 (n_samples,) 整数预测
        y_true: 真实标签, shape=(n_samples,) 整数

    Returns:
        准确率 (0-1)
    """
    if y_pred.ndim == 2:
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = y_pred

    return np.mean(y_pred_class == y_true)


def confusion_matrix(y_pred, y_true, num_classes=None):
    """计算混淆矩阵

    Args:
        y_pred: 预测标签, shape=(n_samples,)
        y_true: 真实标签, shape=(n_samples,)
        num_classes: 类别数

    Returns:
        混淆矩阵, shape=(num_classes, num_classes)
    """
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    if num_classes is None:
        num_classes = max(np.max(y_true) + 1, np.max(y_pred) + 1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def precision_recall_f1(y_pred, y_true, num_classes=None):
    """计算每个类别的精确率、召回率和 F1 分数

    Args:
        y_pred: 预测标签, shape=(n_samples,)
        y_true: 真实标签, shape=(n_samples,)

    Returns:
        precision, recall, f1: 每个类别的值
    """
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    if num_classes is None:
        num_classes = max(np.max(y_true) + 1, np.max(y_pred) + 1)

    cm = confusion_matrix(y_pred, y_true, num_classes)

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0

    return precision, recall, f1


def mean_absolute_error(y_pred, y_true):
    """平均绝对误差"""
    return np.mean(np.abs(y_pred - y_true))


def root_mean_squared_error(y_pred, y_true):
    """均方根误差"""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
