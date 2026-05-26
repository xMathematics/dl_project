"""Sequential 容器 — 按顺序组织多个层"""

from .base import Layer


class Sequential(Layer):
    """顺序模型容器

    将多个层按顺序组合，前向传播依次通过所有层，
    反向传播从最后一层反向遍历。

    Args:
        layers: 层列表
    """

    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers if layers is not None else []
        self._param_counted = False

    def add(self, layer):
        """添加层"""
        self.layers.append(layer)

    def forward(self, x):
        """顺序前向传播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """逆序反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self):
        """设置为所有层的训练模式"""
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        """设置为所有层的评估模式"""
        self.training = False
        for layer in self.layers:
            layer.eval()

    def get_params(self):
        """递归获取所有子层的参数"""
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            for k, v in layer_params.items():
                params[f'{i}.{layer.__class__.__name__}.{k}'] = v
        return params

    def get_grads(self):
        """递归获取所有子层的梯度"""
        grads = {}
        for i, layer in enumerate(self.layers):
            layer_grads = layer.get_grads()
            for k, v in layer_grads.items():
                grads[f'{i}.{layer.__class__.__name__}.{k}'] = v
        return grads

    def zero_grad(self):
        """清零所有子层的梯度"""
        for layer in self.layers:
            layer.zero_grad()

    def num_params(self):
        """计算总参数量"""
        total = 0
        for layer in self.layers:
            total += layer.num_params()
        return total

    def __repr__(self):
        layer_strs = [f'  ({i}): {layer}' for i, layer in enumerate(self.layers)]
        return 'Sequential(\n' + '\n'.join(layer_strs) + '\n)'
