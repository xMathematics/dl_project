"""网络层基类

所有深度学习层的抽象基类。
每个层必须实现 forward() 和 backward() 方法。
"""

# NumPy 提供高效的数值计算和数组操作
import numpy as np


class Layer:
    """所有神经网络层的基类

    每个层需要实现:
    - forward(x): 前向传播 — 计算输出
    - backward(grad): 反向传播 — 计算梯度

    公共属性:
    - params: 可训练参数字典 {name: ndarray}
    - grads:  参数梯度字典 {name: ndarray}
    - cache:  前向传播缓存，供反向传播使用
    - training: 训练模式标志 (True=训练, False=评估)
    """

    def __init__(self):
        # params: 可训练参数，如权重 W 和偏置 b
        # 每个子类在 _init_params 中填充
        self.params = {}
        # grads: 参数梯度，与 params 的键一一对应
        # 在 backward() 中计算，由优化器 step() 使用
        self.grads = {}
        # cache: 前向传播时保存中间结果 (如输入 x)
        # 反向传播时从中读取，避免重复计算
        self.cache = {}
        # training: 控制训练/评估模式
        # True: 训练 (Dropout 生效, BN 使用 batch 统计)
        # False: 评估 (Dropout 关闭, BN 使用运行统计)
        self.training = True

    def forward(self, x):
        """前向传播

        计算图层输出 y = f(x; params)。
        同时将中间值存入 self.cache 供 backward 使用。

        参数:
            x: ndarray, 输入数据, shape=(batch_size, ...)

        返回:
            out: ndarray, 输出数据

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def backward(self, grad):
        """反向传播

        根据上游梯度 grad = dL/d(out) 计算:
        1. 参数梯度: 存入 self.grads (优化器使用)
        2. 输入梯度: dL/dx (返回给上一层)

        参数:
            grad: ndarray, 上游梯度, shape=(batch_size, ...)

        返回:
            dx: ndarray, 对输入的梯度, shape=(batch_size, ...)

        抛出:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def train(self):
        """切换为训练模式

        Dropout: 随机丢弃神经元
        BatchNorm: 使用当前 batch 的均值和方差
        """
        self.training = True

    def eval(self):
        """切换为评估/推理模式

        Dropout: 不再丢弃神经元（全部保留）
        BatchNorm: 使用训练时累积的运行均值和方差
        """
        self.training = False

    def get_params(self):
        """获取所有可训练参数

        返回:
            params: dict, {参数名: ndarray}
        """
        return self.params

    def get_grads(self):
        """获取所有参数的梯度

        返回:
            grads: dict, {参数名: ndarray}
        """
        return self.grads

    def zero_grad(self):
        """将梯度清为零

        通常在每次参数更新后调用，防止梯度累积。
        """
        for k in self.grads:
            # np.zeros_like: 创建与梯度形状相同的全零数组
            self.grads[k] = np.zeros_like(self.grads[k])

    def num_params(self):
        """计算本层的总参数量

        遍历 params 中每个数组，累加所有元素个数。

        返回:
            total: int, 参数总数
        """
        total = 0
        for p in self.params.values():
            # np.prod(p.shape): 计算数组所有维度的乘积 (= 元素个数)
            total += np.prod(p.shape)
        return total

    def __repr__(self):
        """返回层的字符串表示"""
        return f"{self.__class__.__name__}()"
