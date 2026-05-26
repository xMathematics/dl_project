"""激活函数 — 每个激活函数都实现 forward / backward，可像 Layer 一样使用"""

import numpy as np


class _ActivationBase:
    """激活函数基类 — 提供与 Layer 兼容的接口"""

    def get_params(self):
        return {}

    def get_grads(self):
        return {}

    def num_params(self):
        return 0

    def zero_grad(self):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Sigmoid(_ActivationBase):
    """Sigmoid 激活: f(x) = 1 / (1 + exp(-x))"""

    def forward(self, x):
        self.cache = x
        result = 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        return result

    def backward(self, grad):
        x = self.cache
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        return grad * s * (1 - s)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Sigmoid"


class Tanh(_ActivationBase):
    """Tanh 激活: f(x) = tanh(x)"""

    def forward(self, x):
        self.cache = x
        return np.tanh(x)

    def backward(self, grad):
        x = self.cache
        t = np.tanh(x)
        return grad * (1 - t ** 2)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Tanh"


class ReLU(_ActivationBase):
    """ReLU 激活: f(x) = max(0, x)"""

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad):
        x = self.cache
        return grad * (x > 0).astype(float)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ReLU"


class LeakyReLU(_ActivationBase):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x"""

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        x = self.cache
        return grad * np.where(x > 0, 1.0, self.alpha)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"LeakyReLU({self.alpha})"


class ELU(_ActivationBase):
    """ELU 激活: f(x) = x if x > 0 else alpha * (exp(x) - 1)"""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.cache = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return out

    def backward(self, grad):
        x = self.cache
        dx = np.where(x > 0, 1.0, self.alpha * np.exp(x))
        return grad * dx

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"ELU({self.alpha})"


class GELU(_ActivationBase):
    """GELU 激活: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""

    def forward(self, x):
        self.cache = x
        c = np.sqrt(2 / np.pi)
        out = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
        return out

    def backward(self, grad):
        x = self.cache
        c = np.sqrt(2 / np.pi)
        tanh_arg = c * (x + 0.044715 * x ** 3)
        tanh_val = np.tanh(tanh_arg)
        sech2 = 1 - tanh_val ** 2
        dx = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x ** 2)
        return grad * dx

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "GELU"


class Softmax(_ActivationBase):
    """Softmax 激活 (主要用于输出层)"""

    def forward(self, x, axis=-1):
        """Softmax 前向传播

        Args:
            x: 输入, shape=(batch_size, num_classes)
            axis: 计算维度

        Returns:
            概率分布
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max
        exp_x = np.exp(x_shifted)
        self.cache = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return self.cache

    def backward(self, grad):
        """Softmax 反向传播 (通常与 CrossEntropy 结合，单独使用较少)"""
        s = self.cache
        return s * (grad - np.sum(grad * s, axis=-1, keepdims=True))

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Softmax"


class Linear(_ActivationBase):
    """线性激活 (恒等映射)"""

    def forward(self, x):
        self.cache = x
        return x.copy()

    def backward(self, grad):
        return grad

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Linear"
