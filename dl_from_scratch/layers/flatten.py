"""Flatten 层 — 将多维输入展平为 2D"""

import numpy as np
from .base import Layer


class Flatten(Layer):
    """展平层: (batch_size, channels, height, width) -> (batch_size, channels * height * width)"""

    def forward(self, x):
        self.cache['shape'] = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad):
        original_shape = self.cache['shape']
        return grad.reshape(original_shape)

    def __repr__(self):
        return "Flatten()"
