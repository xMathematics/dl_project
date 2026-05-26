"""Transformer 组件 — TransformerBlock & PositionalEncoding"""

import numpy as np
from .base import Layer
from .attention import MultiHeadAttention
from .dense import Dense
from .dropout import Dropout
from .normalization import LayerNormalization
from ..activations.functions import ReLU, GELU


class PositionalEncoding(Layer):
    """位置编码 — 使用正弦/余弦函数 (无需训练参数)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        max_len: 最大序列长度
        d_model: 嵌入维度
    """

    def __init__(self, max_len=5000, d_model=512):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # 预计算位置编码
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]  # (1, max_len, d_model)

    def forward(self, x):
        """前向传播 — 添加位置编码到输入

        Args:
            x: shape=(batch_size, seq_len, d_model)

        Returns:
            shape=(batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]

    def backward(self, grad):
        """位置编码无参数，梯度直接传递"""
        return grad

    def __repr__(self):
        return f"PositionalEncoding(max_len={self.max_len}, d_model={self.d_model})"


class TransformerBlock(Layer):
    """Transformer 编码器块

    结构: Multi-Head Attention → Add & LayerNorm → FFN → Add & LayerNorm

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络隐藏层维度
        dropout_rate: Dropout 概率
        activation: 前馈网络激活函数
        use_causal_mask: 是否使用因果掩码
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout_rate=0.1,
                 activation=None, use_causal_mask=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout_rate = dropout_rate
        self.activation = activation if activation is not None else ReLU()
        self.use_causal_mask = use_causal_mask

        # 子层
        self.attention = MultiHeadAttention(d_model, num_heads, use_causal_mask)
        self.attn_dropout = Dropout(dropout_rate)
        self.attn_norm = LayerNormalization()

        self.ffn_1 = Dense(self.d_ff)
        self.ffn_2 = Dense(d_model)
        self.ffn_dropout = Dropout(dropout_rate)
        self.ffn_norm = LayerNormalization()

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, seq_len, d_model)

        Returns:
            shape=(batch_size, seq_len, d_model)
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_out = self.attention.forward(x)
        attn_out = self.attn_dropout.forward(attn_out)
        x = self.attn_norm.forward(x + attn_out)

        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn_1.forward(x)
        ffn_out = self.activation.forward(ffn_out)
        ffn_out = self.ffn_2.forward(ffn_out)
        ffn_out = self.ffn_dropout.forward(ffn_out)
        x = self.ffn_norm.forward(x + ffn_out)

        return x

    def backward(self, grad):
        """反向传播

        前向: x1 = attn_norm(x + attn(x)) → x2 = ffn_norm(x1 + ffn(x1))
        反向需处理两个残差连接的分支合并。
        """
        # --- 第二个残差块 (FFN) ---
        # d(x1 + ffn_out) = ffn_norm.backward(grad)
        d_2nd_residual = self.ffn_norm.backward(grad)
        # 保存 shortcut 分支梯度: d(x1) from residual
        d_shortcut_2 = d_2nd_residual.copy()
        # FFN 路径反向: d(x1) from FFN path
        grad = self.ffn_dropout.backward(d_2nd_residual)
        grad = self.ffn_2.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.ffn_1.backward(grad)
        # 合并残差: d(x1)_total = d(x1)_ffn + d(x1)_shortcut
        grad = grad + d_shortcut_2

        # --- 第一个残差块 (Attention) ---
        # 保存 shortcut 分支梯度: d(x) from residual
        d_shortcut_1 = grad.copy()
        # Attention 路径反向
        grad = self.attn_norm.backward(grad)
        grad = self.attn_dropout.backward(grad)
        grad = self.attention.backward(grad)
        # 合并残差: d(x)_total = d(x)_attn + d(x)_shortcut
        return grad + d_shortcut_1

    def train(self):
        """训练模式"""
        self.training = True
        self.attention.train()
        self.attn_dropout.train()
        self.attn_norm.train()
        self.ffn_1.train()
        self.ffn_2.train()
        self.ffn_dropout.train()
        self.ffn_norm.train()

    def eval(self):
        """评估模式"""
        self.training = False
        self.attention.eval()
        self.attn_dropout.eval()
        self.attn_norm.eval()
        self.ffn_1.eval()
        self.ffn_2.eval()
        self.ffn_dropout.eval()
        self.ffn_norm.eval()

    def get_params(self):
        params = {}
        for name, sublayer in [
            ('attention', self.attention),
            ('attn_norm', self.attn_norm),
            ('ffn_1', self.ffn_1),
            ('ffn_2', self.ffn_2),
            ('ffn_norm', self.ffn_norm),
        ]:
            for k, v in sublayer.get_params().items():
                params[f'{name}.{k}'] = v
        return params

    def get_grads(self):
        grads = {}
        for name, sublayer in [
            ('attention', self.attention),
            ('attn_norm', self.attn_norm),
            ('ffn_1', self.ffn_1),
            ('ffn_2', self.ffn_2),
            ('ffn_norm', self.ffn_norm),
        ]:
            for k, v in sublayer.get_grads().items():
                grads[f'{name}.{k}'] = v
        return grads

    def zero_grad(self):
        self.attention.zero_grad()
        self.attn_norm.zero_grad()
        self.ffn_1.zero_grad()
        self.ffn_2.zero_grad()
        self.ffn_norm.zero_grad()

    def __repr__(self):
        return (f"TransformerBlock(d_model={self.d_model}, heads={self.num_heads}, "
                f"d_ff={self.d_ff})")
