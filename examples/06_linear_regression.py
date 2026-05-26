"""
示例 6: 线性回归 — 从零实现

学习目标:
1. 理解最小二乘法的闭式解 (正规方程)
2. 理解梯度下降优化
3. 学习 R² 评估指标
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.ml.linear_regression import LinearRegression


def main():
    print("=" * 60)
    print("示例 6: 线性回归 — 房价预测")
    print("=" * 60)

    # 1. 生成模拟数据: y = 3*x1 + 2*x2 + 1 + 噪声
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    true_w = np.array([3.0, 2.0])
    true_b = 1.0
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.5

    print(f"\n真实参数: w = {true_w}, b = {true_b}")

    # 2. 划分训练/测试
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. 闭式解
    print("\n方法1: 正规方程 (闭式解)")
    model1 = LinearRegression(method='closed_form')
    model1.fit(X_train, y_train)
    print(f"  学习参数: w = {model1.coef_}, b = {model1.intercept_:.4f}")
    print(f"  训练 R²: {model1.score(X_train, y_train):.4f}")
    print(f"  测试 R²: {model1.score(X_test, y_test):.4f}")

    # 4. 梯度下降
    print("\n方法2: 梯度下降")
    model2 = LinearRegression(method='gd', lr=0.1, epochs=500)
    model2.fit(X_train, y_train)
    print(f"  学习参数: w = {model2.coef_}, b = {model2.intercept_:.4f}")
    print(f"  训练 R²: {model2.score(X_train, y_train):.4f}")
    print(f"  测试 R²: {model2.score(X_test, y_test):.4f}")

    # 5. 单变量可视化
    print("\n单变量线性回归 (仅使用 x1 预测):")
    X1 = X[:, 0:1]
    X1_train, X1_test = X1[:split], X1[split:]
    model3 = LinearRegression()
    model3.fit(X1_train, y_train)
    print(f"  y = {model3.coef_[0]:.4f} * x1 + {model3.intercept_:.4f}")
    print(f"  测试 R²: {model3.score(X1_test, y_test):.4f} (单变量)")


if __name__ == '__main__':
    main()
