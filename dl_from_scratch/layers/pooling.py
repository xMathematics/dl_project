"""池化层 — MaxPool2D & AvgPool2D"""

import numpy as np
from .base import Layer


class MaxPool2D(Layer):
    """最大池化层

    Args:
        pool_size: 池化窗口大小 (整数或 (h, w))
        stride: 步长 (整数或 (h, w))，默认等于 pool_size
    """

    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, channels, height, width)

        Returns:
            shape=(batch_size, channels, out_h, out_w)
        """
        batch_size, channels, in_h, in_w = x.shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.stride

        out_h = (in_h - p_h) // s_h + 1
        out_w = (in_w - p_w) // s_w + 1

        out = np.zeros((batch_size, channels, out_h, out_w))
        max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                h_end = h_start + p_h
                w_start = j * s_w
                w_end = w_start + p_w

                window = x[:, :, h_start:h_end, w_start:w_end]
                # window shape: (batch_size, channels, p_h, p_w)

                # 找到最大值及其位置
                window_flat = window.reshape(batch_size, channels, -1)
                max_idx_flat = np.argmax(window_flat, axis=-1)

                out[:, :, i, j] = np.max(window, axis=(2, 3))

                # 保存最大值位置用于反向传播
                max_indices[:, :, i, j, 0] = max_idx_flat // (p_w)
                max_indices[:, :, i, j, 1] = max_idx_flat % (p_w)

        self.cache = {
            'x_shape': x.shape,
            'max_indices': max_indices,
            'out_h': out_h,
            'out_w': out_w,
        }
        return out

    def backward(self, grad):
        """反向传播 — 梯度只回传到最大值位置"""
        x_shape = self.cache['x_shape']
        max_indices = self.cache['max_indices']
        out_h = self.cache['out_h']
        out_w = self.cache['out_w']
        batch_size, channels, in_h, in_w = x_shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.stride

        dx = np.zeros(x_shape)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                w_start = j * s_w
                idx_h = max_indices[:, :, i, j, 0]
                idx_w = max_indices[:, :, i, j, 1]

                for b in range(batch_size):
                    for c in range(channels):
                        dx[b, c, h_start + idx_h[b, c], w_start + idx_w[b, c]] += grad[b, c, i, j]

        return dx

    def __repr__(self):
        return f"MaxPool2D(pool={self.pool_size}, stride={self.stride})"


class AvgPool2D(Layer):
    """平均池化层"""

    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

    def forward(self, x):
        batch_size, channels, in_h, in_w = x.shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.stride

        out_h = (in_h - p_h) // s_h + 1
        out_w = (in_w - p_w) // s_w + 1

        out = np.zeros((batch_size, channels, out_h, out_w))
        pool_area = p_h * p_w

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                h_end = h_start + p_h
                w_start = j * s_w
                w_end = w_start + p_w
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        self.cache = {
            'x_shape': x.shape,
            'out_h': out_h,
            'out_w': out_w,
            'pool_area': pool_area,
        }
        return out

    def backward(self, grad):
        x_shape = self.cache['x_shape']
        out_h = self.cache['out_h']
        out_w = self.cache['out_w']
        pool_area = self.cache['pool_area']
        batch_size, channels, in_h, in_w = x_shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.stride

        dx = np.zeros(x_shape)
        grad_expanded = grad / pool_area

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s_h
                h_end = h_start + p_h
                w_start = j * s_w
                w_end = w_start + p_w
                dx[:, :, h_start:h_end, w_start:w_end] += grad_expanded[:, :, i:i+1, j:j+1]

        return dx

    def __repr__(self):
        return f"AvgPool2D(pool={self.pool_size}, stride={self.stride})"
