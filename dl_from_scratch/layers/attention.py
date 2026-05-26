"""注意力机制 — Self-Attention & Multi-Head Attention"""

import numpy as np
from .base import Layer
from ..core.initializers import he_normal


class SelfAttention(Layer):
    """缩放点积自注意力 (Scaled Dot-Product Self-Attention)

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        embed_dim: 输入嵌入维度
        use_causal_mask: 是否使用因果掩码 (防止看到未来)
    """

    def __init__(self, embed_dim, use_causal_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_causal_mask = use_causal_mask
        self._initialized = False

    def _init_params(self):
        self.params['W_q'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_k'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_v'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_o'] = he_normal((self.embed_dim, self.embed_dim))

        for k in ['W_q', 'W_k', 'W_v', 'W_o']:
            self.grads[k] = np.zeros_like(self.params[k])
        self._initialized = True

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, seq_len, embed_dim)

        Returns:
            shape=(batch_size, seq_len, embed_dim)
        """
        if not self._initialized:
            self._init_params()

        batch_size, seq_len, embed_dim = x.shape
        d_k = embed_dim

        # 线性投影
        Q = x @ self.params['W_q']  # (B, S, d)
        K = x @ self.params['W_k']  # (B, S, d)
        V = x @ self.params['W_v']  # (B, S, d)

        # 缩放点积注意力分数
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (B, S, S)

        # 因果掩码 (可选)
        if self.use_causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores - 1e9 * mask[np.newaxis, :, :]

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_shifted = scores - scores_max
        attn_weights = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=-1, keepdims=True)

        # 加权求和
        out = attn_weights @ V  # (B, S, d)

        # 输出投影
        out = out @ self.params['W_o']

        # 缓存反向传播所需数据
        self.cache = {
            'x': x,
            'Q': Q,
            'K': K,
            'V': V,
            'attn_weights': attn_weights,
            'scores': scores,
        }
        return out

    def backward(self, grad):
        """反向传播"""
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attn_weights = self.cache['attn_weights']
        scores = self.cache['scores']
        batch_size, seq_len, d_k = x.shape

        # 输出投影梯度
        d_out = grad @ self.params['W_o'].T  # (B, S, d)
        self.grads['W_o'] = (self.cache['attn_weights'] @ V).transpose(1, 2, 0) @ grad.transpose(1, 0, 2)
        self.grads['W_o'] = self.grads['W_o'].transpose(1, 0, 2).sum(axis=0)
        # 简化版本:
        attn_out = attn_weights @ V  # (B, S, d)
        # dW_o = ∑_batch attn_out^T @ grad
        self.grads['W_o'] = np.sum(
            attn_out.transpose(0, 2, 1) @ grad,
            axis=0
        ).T

        # 重新计算 d_out
        d_out = grad @ self.params['W_o'].T

        # dV = attn_weights^T @ d_out
        dV = attn_weights.transpose(0, 2, 1) @ d_out  # (B, S, d)

        # d(attn_weights) = d_out @ V^T
        d_attn = d_out @ V.transpose(0, 2, 1)  # (B, S, S)

        # Softmax 反向传播
        # d_scores = attn_weights * (d_attn - sum(attn_weights * d_attn, axis=-1, keepdims=True))
        d_scores = attn_weights * (
            d_attn - np.sum(attn_weights * d_attn, axis=-1, keepdims=True)
        )

        # 缩放
        d_scores = d_scores / np.sqrt(d_k)

        # dQ = d_scores @ K, dK = d_scores^T @ Q, dV already computed
        dQ = d_scores @ K
        dK = d_scores.transpose(0, 2, 1) @ Q

        # dW_q = x^T @ dQ, dW_k = x^T @ dK, dW_v = x^T @ dV
        self.grads['W_q'] = np.sum(x.transpose(0, 2, 1) @ dQ, axis=0)
        self.grads['W_k'] = np.sum(x.transpose(0, 2, 1) @ dK, axis=0)
        self.grads['W_v'] = np.sum(x.transpose(0, 2, 1) @ dV, axis=0)

        # 对输入的梯度
        dx = (dQ @ self.params['W_q'].T +
              dK @ self.params['W_k'].T +
              dV @ self.params['W_v'].T)

        return dx

    def __repr__(self):
        return f"SelfAttention(embed_dim={self.embed_dim}, causal={self.use_causal_mask})"


class MultiHeadAttention(Layer):
    """多头注意力 (Multi-Head Attention)

    将 Q, K, V 投影到多个子空间，分别计算注意力后拼接。

    Args:
        embed_dim: 输入/输出嵌入维度
        num_heads: 注意力头数
        use_causal_mask: 是否使用因果掩码
    """

    def __init__(self, embed_dim, num_heads=8, use_causal_mask=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self._initialized = False

    def _init_params(self):
        # Q, K, V 投影 (每个头的投影合并)
        self.params['W_q'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_k'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_v'] = he_normal((self.embed_dim, self.embed_dim))
        self.params['W_o'] = he_normal((self.embed_dim, self.embed_dim))

        for k in ['W_q', 'W_k', 'W_v', 'W_o']:
            self.grads[k] = np.zeros_like(self.params[k])
        self._initialized = True

    def _split_heads(self, x):
        """拆分多头: (B, S, D) -> (B, H, S, D/H)"""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        """合并多头: (B, H, S, D/H) -> (B, S, D)"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x):
        """前向传播

        Args:
            x: shape=(batch_size, seq_len, embed_dim)

        Returns:
            shape=(batch_size, seq_len, embed_dim)
        """
        if not self._initialized:
            self._init_params()

        batch_size, seq_len, _ = x.shape

        # 线性投影 + 拆分多头
        Q = self._split_heads(x @ self.params['W_q'])  # (B, H, S, D/H)
        K = self._split_heads(x @ self.params['W_k'])
        V = self._split_heads(x @ self.params['W_v'])

        # 缩放点积注意力
        scale = np.sqrt(self.head_dim)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale  # (B, H, S, S)

        # 因果掩码
        if self.use_causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores - 1e9 * mask[np.newaxis, np.newaxis, :, :]

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_shifted = scores - scores_max
        attn_weights = np.exp(scores_shifted) / (np.sum(np.exp(scores_shifted), axis=-1, keepdims=True) + 1e-10)

        # 加权求和
        context = attn_weights @ V  # (B, H, S, D/H)

        # 合并多头 + 输出投影
        context = self._combine_heads(context)  # (B, S, D)
        out = context @ self.params['W_o']

        # 缓存
        self.cache = {
            'x': x,
            'Q': Q,
            'K': K,
            'V': V,
            'attn_weights': attn_weights,
            'scores': scores,
            'context': context,
        }
        return out

    def backward(self, grad):
        """反向传播"""
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attn_weights = self.cache['attn_weights']
        context = self.cache['context']
        batch_size, seq_len, _ = x.shape

        # dW_o
        self.grads['W_o'] = np.sum(
            context.transpose(0, 2, 1) @ grad,
            axis=0
        )

        # d_context
        d_context = grad @ self.params['W_o'].T
        d_context = d_context.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        d_context = d_context.transpose(0, 2, 1, 3)  # (B, H, S, D/H)

        # dV
        dV = attn_weights.transpose(0, 1, 3, 2) @ d_context

        # d_attn_weights
        d_attn = d_context @ V.transpose(0, 1, 3, 2)

        # Softmax 反向传播
        d_scores = attn_weights * (
            d_attn - np.sum(attn_weights * d_attn, axis=-1, keepdims=True)
        )
        d_scores = d_scores / np.sqrt(self.head_dim)

        # dQ, dK
        dQ = d_scores @ K
        dK = d_scores.transpose(0, 1, 3, 2) @ Q

        # 合并多头
        dQ = self._combine_heads(dQ)
        dK = self._combine_heads(dK)
        dV = self._combine_heads(dV)

        # 投影梯度
        self.grads['W_q'] = np.sum(x.transpose(0, 2, 1) @ dQ, axis=0)
        self.grads['W_k'] = np.sum(x.transpose(0, 2, 1) @ dK, axis=0)
        self.grads['W_v'] = np.sum(x.transpose(0, 2, 1) @ dV, axis=0)

        # 输入梯度
        dx = (dQ @ self.params['W_q'].T +
              dK @ self.params['W_k'].T +
              dV @ self.params['W_v'].T)

        return dx

    def __repr__(self):
        return f"MultiHeadAttention(embed_dim={self.embed_dim}, heads={self.num_heads})"
