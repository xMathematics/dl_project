"""卷积层 (Conv2D) — 纯 NumPy 实现"""

import numpy as np
from .base import Layer
from ..core.initializers import he_normal, zeros


class Conv2D(Layer):
    """二维卷积层

    Args:
        out_channels: 输出通道数 (卷积核数量)
        kernel_size: 卷积核大小 (整数或 (h, w))
        stride: 步长 (整数或 (h, w))
        padding: 填充大小 (整数或 (h, w))
        use_bias: 是否使用偏置
    """

    def __init__(self, out_channels, kernel_size, stride=1, padding=0,
                 use_bias=True, weight_initializer=he_normal, bias_initializer=zeros):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self._initialized = False

    def _init_params(self, in_channels):
        """延迟初始化卷积核"""
        k_h, k_w = self.kernel_size
        weight_shape = (self.out_channels, in_channels, k_h, k_w)
        self.params['W'] = self.weight_initializer(weight_shape)
        if self.use_bias:
            self.params['b'] = self.bias_initializer((1, self.out_channels, 1, 1))
        self.grads['W'] = np.zeros_like(self.params['W'])
        if self.use_bias:
            self.grads['b'] = np.zeros_like(self.params['b'])
        self._initialized = True

    def _pad_input(self, x):
        """对输入进行填充"""
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, in_channels, height, width)

        Returns:
            shape=(batch_size, out_channels, out_h, out_w)
        """
        if not self._initialized:
            self._init_params(x.shape[1])

        batch_size, in_channels, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        # 计算输出尺寸
        out_h = (in_h + 2 * p_h - k_h) // s_h + 1
        out_w = (in_w + 2 * p_w - k_w) // s_w + 1

        # 填充输入
        x_padded = self._pad_input(x)
        self.cache['x_padded'] = x_padded
        self.cache['x_shape'] = x.shape

        W = self.params['W']  # (out_c, in_c, k_h, k_w)

        # im2col: 将卷积操作转化为矩阵乘法
        col = self._im2col(x_padded, k_h, k_w, s_h, s_w)
        # col shape: (batch_size * out_h * out_w, in_c * k_h * k_w)

        # W_flat: (out_c, in_c * k_h * k_w)
        W_flat = W.reshape(self.out_channels, -1)

        # out: (batch_size * out_h * out_w, out_c)
        out = col @ W_flat.T

        # 重塑为输出形状
        out = out.reshape(batch_size, out_h, out_w, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (N, C, H, W)

        if self.use_bias:
            out += self.params['b']

        self.cache['col'] = col
        return out

    def backward(self, grad):
        """反向传播

        Args:
            grad: 上游梯度, shape=(batch_size, out_channels, out_h, out_w)

        Returns:
            对输入的梯度, shape=(batch_size, in_channels, in_h, in_w)
        """
        x_padded = self.cache['x_padded']
        x_shape = self.cache['x_shape']
        col = self.cache['col']
        batch_size, in_channels, in_h, in_w = x_shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        W = self.params['W']  # (out_c, in_c, k_h, k_w)
        out_h, out_w = grad.shape[2], grad.shape[3]

        # 梯度重塑
        grad_reshaped = grad.transpose(0, 2, 3, 1)  # (N, H, W, out_c)
        grad_reshaped = grad_reshaped.reshape(-1, self.out_channels)  # (N*H*W, out_c)

        # dL/dW: 使用 im2col 的 col 和 grad
        # W_grad = col^T @ grad_reshaped  -> (in_c*k_h*k_w, out_c)
        dW_flat = col.T @ grad_reshaped  # (in_c*k_h*k_w, out_c)
        self.grads['W'] = dW_flat.T.reshape(W.shape)  # (out_c, in_c, k_h, k_w)

        if self.use_bias:
            self.grads['b'] = np.sum(grad, axis=(0, 2, 3), keepdims=True)

        # dL/dx: col2im
        # dx_col = grad_reshaped @ W_flat  -> (N*H*W, in_c*k_h*k_w)
        W_flat = W.reshape(self.out_channels, -1)
        dx_col = grad_reshaped @ W_flat  # (N*H*W, in_c*k_h*k_w)

        # 将 col 恢复为原始输入
        dx_padded = self._col2im(dx_col, x_padded.shape, k_h, k_w, s_h, s_w)

        # 裁剪填充部分
        if p_h > 0 or p_w > 0:
            dx = dx_padded[:, :, p_h:p_h + in_h, p_w:p_w + in_w]
        else:
            dx = dx_padded

        return dx

    def _im2col(self, x, k_h, k_w, s_h, s_w):
        """将图像转换为矩阵列 (im2col)"""
        batch_size, channels, in_h, in_w = x.shape
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1

        col = np.zeros((batch_size, channels, k_h, k_w, out_h, out_w))

        for i in range(k_h):
            i_end = i + s_h * out_h
            for j in range(k_w):
                j_end = j + s_w * out_w
                col[:, :, i, j, :, :] = x[:, :, i:i_end:s_h, j:j_end:s_w]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
        return col

    def _col2im(self, col, x_shape, k_h, k_w, s_h, s_w):
        """将矩阵列恢复为图像 (col2im)"""
        batch_size, channels, in_h, in_w = x_shape
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1

        col_reshaped = col.reshape(batch_size, out_h, out_w, channels, k_h, k_w)
        col_reshaped = col_reshaped.transpose(0, 3, 4, 5, 1, 2)

        x_restored = np.zeros(x_shape)
        count = np.zeros(x_shape)

        for i in range(k_h):
            i_end = i + s_h * out_h
            for j in range(k_w):
                j_end = j + s_w * out_w
                x_restored[:, :, i:i_end:s_h, j:j_end:s_w] += col_reshaped[:, :, i, j, :, :]
                count[:, :, i:i_end:s_h, j:j_end:s_w] += 1

        # 避免除零
        count = np.maximum(count, 1)
        return x_restored / count

    def __repr__(self):
        k_h, k_w = self.kernel_size
        return f"Conv2D(out_channels={self.out_channels}, kernel=({k_h},{k_w}), stride={self.stride})"
