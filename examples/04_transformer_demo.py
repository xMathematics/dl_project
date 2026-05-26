"""
示例 4: Transformer — 从零实现

学习目标:
1. 理解自注意力机制: Q, K, V 的计算
2. 理解多头注意力
3. 理解位置编码
4. 理解 TransformerBlock 的结构

本示例构建一个迷你 Transformer，在简单的序列任务上演示。
任务: 预测序列中下一个数字 (简单的序列建模)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dl_from_scratch.layers.dense import Dense
from dl_from_scratch.layers.embedding import Embedding
from dl_from_scratch.layers.transformer import TransformerBlock, PositionalEncoding
from dl_from_scratch.layers.sequential import Sequential
from dl_from_scratch.activations.functions import ReLU
from dl_from_scratch.losses.functions import CrossEntropyLoss
from dl_from_scratch.optimizers.sgd import Adam
from dl_from_scratch.models.model import Model
from dl_from_scratch.utils.metrics import accuracy
from dl_from_scratch.utils.data import to_one_hot


def generate_sequence_data(num_samples=500, seq_len=8, vocab_size=10):
    """
    生成序列数据: 预测序列中的下一个数字
    例如: [3, 1, 4, 1, 5, 9, 2, 6] -> 下一个是 5 (圆周率数字)

    简单规则: 如果前一个数 < 5, 输出 1; 否则输出 0
    """
    np.random.seed(42)
    X = np.random.randint(0, vocab_size, size=(num_samples, seq_len))
    # 简单的下一位规则: 基于前一位
    y = np.where(X[:, -1] < 5, 1, 0)
    return X, y


def main():
    print("=" * 60)
    print("示例 4: Transformer — 序列建模")
    print("=" * 60)

    # 超参数 (迷你 Transformer)
    vocab_size = 10      # 词汇表大小
    seq_len = 8          # 序列长度
    d_model = 32         # 模型维度
    num_heads = 4        # 注意力头数 (必须能整除 d_model)
    d_ff = 64            # 前馈网络隐藏层维度
    num_blocks = 2       # Transformer 块数
    batch_size = 32
    epochs = 30

    print(f"\n超参数:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {num_heads}")
    print(f"  FFN 维度: {d_ff}")
    print(f"  Transformer 块数: {num_blocks}")

    # 1. 生成数据
    X, y = generate_sequence_data(num_samples=1000, seq_len=seq_len)
    y_one_hot = to_one_hot(y, 2)

    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_one_hot[:split], y_one_hot[split:]
    y_test_labels = y[split:]

    print(f"\n数据: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    print(f"示例序列: {X[0]}")
    print(f"示例标签 (下一位): {y[0]}")

    # 2. 构建 Transformer 编码器 (输出 (batch, seq, d_model))
    encoder_layers = [
        Embedding(vocab_size=vocab_size, embedding_dim=d_model),
        PositionalEncoding(max_len=seq_len, d_model=d_model),
    ]
    for _ in range(num_blocks):
        encoder_layers.append(TransformerBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff,
            dropout_rate=0.1, activation=ReLU(), use_causal_mask=False,
        ))
    encoder = Sequential(encoder_layers)

    # 分类头 (只取序列最后一个位置的输出)
    classifier = Dense(units=2)

    print(f"\nTransformer 编码器结构:\n{encoder}")
    print(f"分类头: {classifier}")
    total = encoder.num_params() + classifier.num_params()
    print(f"总参数量: {total:,}")

    # 3. 自定义训练 (因为需要取最后一个位置)
    model = Model(encoder)
    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=Adam(lr=0.001)
    )

    print("\n训练 Transformer...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        # 打乱
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 前向: 编码器输出 (batch, seq, d_model)
            model.train()
            enc_out = encoder.forward(X_batch)

            # 取最后一个位置 -> (batch, d_model)
            last_out = enc_out[:, -1, :]

            # 分类头
            logits = classifier.forward(last_out)

            # 损失
            loss = model.loss_fn.forward(logits, y_batch)
            epoch_loss += loss
            num_batches += 1

            # 反向
            grad = model.loss_fn.backward()
            grad = classifier.backward(grad)

            # 反向通过编码器 (需要将梯度 reshape 回 3D)
            grad_3d = np.zeros_like(enc_out)
            grad_3d[:, -1, :] = grad
            encoder.backward(grad_3d)

            # 更新参数
            params = {**encoder.get_params(), **classifier.get_params()}
            grads = {**encoder.get_grads(), **classifier.get_grads()}
            model.optimizer.step(params, grads)

        avg_loss = epoch_loss / num_batches

        # 验证
        model.eval()
        enc_val = encoder.forward(X_test)
        val_logits = classifier.forward(enc_val[:, -1, :])
        val_loss = model.loss_fn.forward(val_logits, y_test)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss: {avg_loss:.6f} | val_loss: {val_loss:.6f}")

    # 5. 评估
    model.eval()
    enc_test = encoder.forward(X_test)
    test_logits = classifier.forward(enc_test[:, -1, :])
    test_acc = accuracy(test_logits, y_test_labels)
    print(f"\n测试准确率: {test_acc:.2%}")

    # 6. 展示注意力机制的核心思想
    print("\n" + "=" * 60)
    print("自注意力机制解释:")
    print("=" * 60)
    print("""
    自注意力核心公式:
        Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

    步骤:
    1. Q (Query) 与 K (Key) 的点积计算相似度分数
    2. 除以 √d_k 缩放，防止梯度消失
    3. Softmax 归一化为注意力权重
    4. 用权重对 V (Value) 加权求和

    多头注意力: 将 Q, K, V 投影到多个子空间，
    让模型在不同位置关注不同方面的信息。
    """)

    # 演示注意力
    print("示例: 序列 '[3, 1, 4, 1, 5, 9, 2, 6]' 通过 Embedding + PositionalEncoding:")
    sample = X[0:1]  # (1, seq_len)
    emb_out = encoder.layers[0].forward(sample)
    pos_out = encoder.layers[1].forward(emb_out)
    print(f"  Embedding 输出形状: {emb_out.shape}")
    print(f"  位置编码后形状: {pos_out.shape}")
    print(f"  位置编码前: {emb_out[0, 0, :4]}...")
    print(f"  位置编码后: {pos_out[0, 0, :4]}...")


if __name__ == '__main__':
    main()
