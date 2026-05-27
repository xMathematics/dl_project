"""激活函数

激活函数为神经网络引入非线性，使网络能学习复杂的模式。
如果没有激活函数，多层网络等价于单层线性变换。

每个激活函数都实现 forward/backward 接口，可像 Layer 一样在 Sequential 中使用。
所有激活函数都没有可训练参数 (但与 Layer 兼容 get_params/num_params 等接口)。
"""

# NumPy 提供高效的逐元素数学运算
import numpy as np


class _ActivationBase:
    """激活函数基类

    提供与 Layer 兼容的接口，使激活函数可以在 Sequential 中与层混合使用。
    激活函数没有可训练参数，所有方法返回空值。
    """

    def get_params(self):
        """激活函数没有可训练参数，返回空字典"""
        return {}

    def get_grads(self):
        """激活函数没有参数梯度，返回空字典"""
        return {}

    def num_params(self):
        """激活函数不占用参数"""
        return 0

    def zero_grad(self):
        """无梯度需要清零"""
        pass

    def train(self):
        """激活函数在训练/评估模式下行为一致"""
        self.training = True

    def eval(self):
        self.training = False


class Sigmoid(_ActivationBase):
    """Sigmoid 激活函数

    数学公式: f(x) = 1 / (1 + e^(-x))

    导数:     f'(x) = f(x) * (1 - f(x))

    值域: (0, 1)，适合二分类输出层

    缺点:
    - 输出不是零中心的 (导致梯度更新效率低)
    - 输入过大/过小时梯度饱和 (梯度消失)
    """

    def forward(self, x):
        """Sigmoid 前向传播

        np.clip(x, -100, 100): 将 x 限制在 [-100, 100] 区间
          - exp(-(-100)) = exp(100) 会溢出为 inf
          - exp(-100) = 3.7e-44 不会下溢
          所以 clip 防止 exp 溢出

        参数:
            x: ndarray, 输入（任意实数）

        返回:
            result: ndarray, 值在 (0, 1) 区间
        """
        # 保存输入到 cache，供 backward 使用
        self.cache = x
        # sigmoid = 1 / (1 + exp(-x))
        # np.exp: 计算 e^z 指数函数
        # np.clip: 限制范围防止数值溢出
        result = 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        return result

    def backward(self, grad):
        """Sigmoid 反向传播

        导数: dsigmoid/dx = s * (1 - s)
        链式法则: dL/dx = dL/ds * ds/dx = grad * s * (1 - s)

        参数:
            grad: ndarray, 上游梯度 dL/d(out)

        返回:
            dx: ndarray, 下游梯度 dL/dx
        """
        # 从 cache 读取前向传播时的输入
        x = self.cache
        # 重新计算 sigmoid 值 (也可以 cache 起来，但重算更省内存)
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        # sigmoid 的导数: s * (1 - s)
        # grad * 导数: 链式法则
        return grad * s * (1 - s)

    def __call__(self, x):
        """使对象可调用: sigmoid(x) 等价于 sigmoid.forward(x)"""
        return self.forward(x)

    def __repr__(self):
        return "Sigmoid"


class Tanh(_ActivationBase):
    """Tanh 双曲正切激活函数

    数学公式: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    导数:     f'(x) = 1 - f(x)²

    值域: (-1, 1)，零中心输出 (优于 Sigmoid)

    与 Sigmoid 的关系: tanh(x) = 2*sigmoid(2x) - 1
    """

    def forward(self, x):
        """Tanh 前向传播

        参数:
            x: ndarray, 输入

        返回:
            ndarray, 值在 (-1, 1) 区间
        """
        # 保存输入用于反向传播
        self.cache = x
        # np.tanh(x): NumPy 内置的双曲正切函数，高效且数值稳定
        return np.tanh(x)

    def backward(self, grad):
        """Tanh 反向传播

        导数: d(tanh)/dx = 1 - tanh(x)² = sech²(x)

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        # 重新计算 tanh(x) (从 cache 读 x 后重算)
        x = self.cache
        t = np.tanh(x)
        # 链式法则: grad * (1 - t²)
        # (1 - t ** 2) 即 sech²(x)，是 tanh 的导数
        return grad * (1 - t ** 2)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Tanh"


class ReLU(_ActivationBase):
    """ReLU 激活 (Rectified Linear Unit)

    数学公式: f(x) = max(0, x)

    导数:     f'(x) = 1 if x > 0 else 0

    优点:
    - 计算简单 (只需比较)
    - 缓解梯度消失 (正区间梯度恒为 1)
    - 稀疏激活 (一半神经元输出 0)

    缺点: 死亡 ReLU (神经元输出恒为 0 后无法恢复)
    """

    def forward(self, x):
        """ReLU 前向传播

        np.maximum(0, x): 逐元素取 0 和 x 的最大值
        负数全部变为 0，正数保持不变。

        参数:
            x: ndarray, 输入 (任意实数)

        返回:
            ndarray: max(0, x)
        """
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad):
        """ReLU 反向传播

        导数: x > 0 时梯度为 1，否则为 0

        (x > 0).astype(float): 将布尔掩码转换为 0.0/1.0
        grad * 掩码: 只对正输入的梯度进行传播

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        x = self.cache
        # x > 0: 布尔数组，True 表示该位置 x > 0
        # .astype(float): 将 True/False 转换为 1.0/0.0
        return grad * (x > 0).astype(float)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ReLU"


class LeakyReLU(_ActivationBase):
    """Leaky ReLU 激活 — ReLU 的改进版本

    数学公式: f(x) = x if x > 0 else α * x

    相比 ReLU 的改进: 负半轴不再为 0，而是给一个很小的斜率 α
    解决了"死亡 ReLU"问题。

    参数:
        alpha: float, 负半轴的斜率 (默认 0.01)
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """Leaky ReLU 前向传播

        np.where(condition, x, y): 根据条件从 x 或 y 中选择
        如果 x > 0，输出 x；否则输出 alpha * x

        参数:
            x: ndarray, 输入

        返回:
            ndarray: Leaky ReLU 输出
        """
        self.cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        """Leaky ReLU 反向传播

        导数: x > 0 时为 1，否则为 alpha

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        x = self.cache
        # np.where(x > 0, 1.0, self.alpha): 正区间导数为 1，负区间为 alpha
        return grad * np.where(x > 0, 1.0, self.alpha)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"LeakyReLU({self.alpha})"


class ELU(_ActivationBase):
    """ELU 激活 (Exponential Linear Unit)

    数学公式: f(x) = x if x > 0 else α * (e^x - 1)

    特点:
    - 正半轴同 ReLU
    - 负半轴指数趋近 -α (输出均值接近 0，加速收敛)
    - 在 x=0 处可导 (光滑过渡)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """ELU 前向传播

        x > 0: 输出 x
        x <= 0: 输出 α * (e^x - 1)
        np.exp(x): 计算 e^x

        参数:
            x: ndarray, 输入

        返回:
            ndarray: ELU 输出
        """
        self.cache = x
        # np.where 根据条件选择: 正区间 x，负区间 α*(e^x-1)
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return out

    def backward(self, grad):
        """ELU 反向传播

        导数: x > 0 时为 1，否则为 α * e^x

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        x = self.cache
        # 正区间导数为 1，负区间导数为 α * e^x
        dx = np.where(x > 0, 1.0, self.alpha * np.exp(x))
        return grad * dx

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"ELU({self.alpha})"


class GELU(_ActivationBase):
    """GELU 激活 (Gaussian Error Linear Unit)

    数学近似: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    特点:
    - 结合了 ReLU 和 dropout 的思想
    - 根据输入值的大小"随机"决定是否激活
    - 在 BERT/GPT 等 Transformer 模型中广泛使用
    - 光滑可导，x≈0 时接近 0.5x
    """

    def forward(self, x):
        """GELU 前向传播 (使用 tanh 近似)

        参数:
            x: ndarray, 输入

        返回:
            ndarray: GELU 输出
        """
        self.cache = x
        # c = √(2/π) ≈ 0.7979
        c = np.sqrt(2 / np.pi)
        # GELU 的 tanh 近似公式
        out = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
        return out

    def backward(self, grad):
        """GELU 反向传播

        注意: 这是 GELU 近似公式的导数，通过链式法则计算。

        参数:
            grad: ndarray, 上游梯度

        返回:
            dx: ndarray, 下游梯度
        """
        x = self.cache
        c = np.sqrt(2 / np.pi)
        # tanh 参数: c * (x + 0.044715 * x³)
        tanh_arg = c * (x + 0.044715 * x ** 3)
        tanh_val = np.tanh(tanh_arg)
        # sech² = 1 - tanh² (tanh 的导数)
        sech2 = 1 - tanh_val ** 2
        # GELU 的导数 (复合函数求导)
        dx = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x ** 2)
        return grad * dx

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "GELU"


class Softmax(_ActivationBase):
    """Softmax 激活函数 (多分类输出层)

    数学公式: softmax(x_i) = e^(x_i) / Σ_j e^(x_j)

    将任意实数向量转换为概率分布:
    - 所有输出 > 0
    - 所有输出之和 = 1

    常用于多分类问题的最后一层。
    """

    def forward(self, x, axis=-1):
        """Softmax 前向传播 (数值稳定版本)

        数值稳定技巧:
        1. 先减去最大值: e^(x_i - max(x)) / Σ e^(x_j - max(x))
        2. 防止 exp 过大溢出

        参数:
            x: ndarray, shape=(batch_size, num_classes)，logits
            axis: int, 在哪个维度上计算 (默认 -1 即最后一维)

        返回:
            probs: ndarray, shape=(batch_size, num_classes)，概率分布
        """
        # 减去最大值: 确保最大指数为 e^0 = 1，不会溢出
        # np.max(x, axis=axis, keepdims=True): 沿 axis 取最大值，保持维度
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max

        # 计算稳定后的指数
        exp_x = np.exp(x_shifted)

        # 归一化为概率: e^(x_i) / Σe^(x_j)
        # np.sum(exp_x, axis=axis, keepdims=True): 求和，保持维度以广播
        self.cache = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return self.cache

    def backward(self, grad):
        """Softmax 反向传播

        Softmax 的雅可比矩阵: ∂s_i/∂z_j = s_i * (δ_ij - s_j)
        其中 δ_ij 是克罗内克 δ (i=j 时为 1，否则为 0)

        向量化实现: dz = s * (grad - Σ(grad * s))

        参数:
            grad: ndarray, 上游梯度 (来自损失函数)

        返回:
            dx: ndarray, 下游梯度
        """
        s = self.cache
        # s * (grad - sum(grad * s)): 向量化的雅可比乘法
        # np.sum(grad * s, axis=-1, keepdims=True): 对每行求 Σ(grad_i * s_i)
        return s * (grad - np.sum(grad * s, axis=-1, keepdims=True))

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Softmax"


class Linear(_ActivationBase):
    """线性激活 (恒等映射 / Identity)

    f(x) = x

    就是什么都不做，直接输出输入。
    通常不需要显式使用，但为了框架一致性而保留。
    """

    def forward(self, x):
        """线性激活前向传播: 直接返回输入的副本

        x.copy(): 返回输入数组的深拷贝，防止外部修改影响 cache
        """
        self.cache = x
        return x.copy()

    def backward(self, grad):
        """线性激活反向传播: 梯度直接通过"""
        return grad

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Linear"
