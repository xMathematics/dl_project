"""
示例 5: 注意力可视化 — 深入理解自注意力机制

学习目标:
1. 可视化注意力权重矩阵
2. 理解 Q, K, V 的相互作用
3. 观察因果掩码的影响

不使用训练，直接观察随机初始化的注意力模式。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.layers.attention import SelfAttention, MultiHeadAttention
from dl_from_scratch.layers.transformer import PositionalEncoding
from dl_from_scratch.layers.embedding import Embedding


def visualize_attention_weights(attn_weights, title="Attention Weights"):
    """可视化注意力权重矩阵"""
    try:
        import matplotlib.pyplot as plt

        if attn_weights.ndim == 3:
            # (seq_len, seq_len) 或 (1, seq_len, seq_len)
            if attn_weights.shape[0] == 1:
                attn_weights = attn_weights[0]
            plt.figure(figsize=(6, 5))
            plt.imshow(attn_weights, cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.title(title)
            plt.tight_layout()
            plt.show()

        elif attn_weights.ndim == 4:
            # (1, num_heads, seq_len, seq_len)
            num_heads = attn_weights.shape[1]
            seq_len = attn_weights.shape[2]
            fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
            if num_heads == 1:
                axes = [axes]
            for h in range(num_heads):
                im = axes[h].imshow(attn_weights[0, h], cmap='viridis', aspect='auto')
                axes[h].set_title(f'Head {h + 1}')
                axes[h].set_xlabel('Key Position')
                axes[h].set_ylabel('Query Position')
                plt.colorbar(im, ax=axes[h])
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
    except ImportError:
        print("(matplotlib 未安装，打印注意力权重矩阵代替)")
        if attn_weights.ndim == 4:
            for h in range(attn_weights.shape[1]):
                print(f"\nHead {h + 1}:")
                print(np.round(attn_weights[0, h], 3))
        else:
            print(np.round(attn_weights.squeeze(), 3))


def main():
    print("=" * 60)
    print("示例 5: 注意力可视化")
    print("=" * 60)

    # 超参数
    seq_len = 8
    d_model = 16
    num_heads = 4

    # 创建模拟输入: 8 个 token 的序列
    # 模拟 "I love deep learning" 的嵌入表示
    np.random.seed(42)
    x = np.random.randn(1, seq_len, d_model).astype(float)

    print(f"输入形状: {x.shape}")
    print(f"输入 (前4维):")
    for i in range(seq_len):
        print(f"  token {i}: {x[0, i, :4].round(3)}")

    # ========== 1. 自注意力 ==========
    print("\n" + "=" * 60)
    print("1. 缩放点积自注意力 (Scaled Dot-Product Attention)")
    print("=" * 60)

    attention = SelfAttention(embed_dim=d_model)
    output = attention.forward(x)
    attn_weights = attention.cache['attn_weights']

    print(f"\n注意力权重矩阵形状: {attn_weights.shape}")
    print("\n每行是 Query 对其他 Key 位置的注意力分布 (行和为 1):")
    visualize_attention_weights(attn_weights, "Self-Attention Weights")

    # 打印每行最大的注意力位置
    print("\n每个 Query 最关注的 Key 位置:")
    for q_idx in range(seq_len):
        top_k_idx = np.argsort(attn_weights[0, q_idx])[-3:][::-1]
        top_k_val = attn_weights[0, q_idx, top_k_idx]
        print(f"  Query[{q_idx}] 关注: ", end="")
        for k, v in zip(top_k_idx, top_k_val):
            print(f"Key[{k}]({v:.3f}) ", end="")
        print()

    # ========== 2. 因果自注意力 ==========
    print("\n" + "=" * 60)
    print("2. 因果自注意力 (Causal Self-Attention)")
    print("=" * 60)

    causal_attention = SelfAttention(embed_dim=d_model, use_causal_mask=True)
    output_causal = causal_attention.forward(x)
    causal_weights = causal_attention.cache['attn_weights']

    print("\n因果注意力权重 (每个 Query 只能看到它自己和之前的 Key):")
    visualize_attention_weights(causal_weights, "Causal Self-Attention (Masked)")

    print("\n因果掩码效果: 右上三角被遮罩 (< 0.01 表示被掩码)")
    for q in range(seq_len):
        for k in range(seq_len):
            if k > q:
                assert causal_weights[0, q, k] < 0.01, "被掩码的位置应该接近 0"
    print("  ✓ 所有未来位置的注意力权重都接近 0")

    # ========== 3. 多头注意力 ==========
    print("\n" + "=" * 60)
    print("3. 多头注意力 (Multi-Head Attention)")
    print("=" * 60)

    mha = MultiHeadAttention(embed_dim=d_model, num_heads=num_heads)
    output_mha = mha.forward(x)
    mha_weights = mha.cache['attn_weights']

    print(f"\n多头注意力权重形状: {mha_weights.shape} (batch, heads, seq, seq)")
    print(f"每个头独立学习不同的注意力模式:")
    visualize_attention_weights(mha_weights, "Multi-Head Attention Weights")

    # 显示每个头的差异
    print("\n各头的注意力分布差异 (KL 散度):")
    for h1 in range(num_heads):
        for h2 in range(h1 + 1, num_heads):
            avg_w1 = mha_weights[0, h1].mean(axis=0)
            avg_w2 = mha_weights[0, h2].mean(axis=0)
            kl = np.sum(avg_w1 * np.log(avg_w1 / (avg_w2 + 1e-10) + 1e-10))
            print(f"  Head {h1+1} vs Head {h2+1}: KL = {kl:.4f}")

    # ========== 4. 位置编码的影响 ==========
    print("\n" + "=" * 60)
    print("4. 位置编码 (Positional Encoding)")
    print("=" * 60)

    pos_encoding = PositionalEncoding(max_len=seq_len, d_model=d_model)
    x_with_pos = pos_encoding.forward(x)

    print(f"\n位置编码形状: {pos_encoding.pe.shape}")
    print(f"位置编码矩阵 (前4维):")
    for pos in range(seq_len):
        print(f"  pos {pos}: {pos_encoding.pe[0, pos, :4].round(3)}")

    # 计算加入位置编码后的注意力变化
    output_pos = attention.forward(x_with_pos)
    pos_attn_weights = attention.cache['attn_weights']
    diff = np.abs(pos_attn_weights - attn_weights).mean()
    print(f"\n加入位置编码后注意力权重的平均变化: {diff:.4f}")
    print("位置编码为模型提供了序列位置信息，使注意力能够感知'位置'关系")


if __name__ == '__main__':
    main()
