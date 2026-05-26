"""
单元测试 — 验证所有模块的正确性

运行方式: python -m pytest tests/test_all.py -v
       或 python tests/test_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ===========================
# 导入所有模块
# ===========================
from dl_from_scratch.layers.dense import Dense
from dl_from_scratch.layers.convolution import Conv2D
from dl_from_scratch.layers.pooling import MaxPool2D, AvgPool2D
from dl_from_scratch.layers.flatten import Flatten
from dl_from_scratch.layers.dropout import Dropout
from dl_from_scratch.layers.normalization import BatchNormalization, LayerNormalization
from dl_from_scratch.layers.embedding import Embedding
from dl_from_scratch.layers.attention import SelfAttention, MultiHeadAttention
from dl_from_scratch.layers.transformer import TransformerBlock, PositionalEncoding
from dl_from_scratch.layers.sequential import Sequential
from dl_from_scratch.activations.functions import Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, ELU, GELU
from dl_from_scratch.losses.functions import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, L1Loss
from dl_from_scratch.optimizers.sgd import SGD, SGDWithMomentum, Adam, RMSprop
from dl_from_scratch.models.model import Model
from dl_from_scratch.utils.data import make_moons, to_one_hot, batch_iterator, train_test_split
from dl_from_scratch.utils.metrics import accuracy, confusion_matrix


def test_dense_forward_backward():
    """测试全连接层"""
    layer = Dense(units=3)
    x = np.random.randn(4, 5)
    out = layer.forward(x)
    assert out.shape == (4, 3), f"期望 (4,3), 得到 {out.shape}"

    grad = np.random.randn(4, 3)
    dx = layer.backward(grad)
    assert dx.shape == (4, 5), f"期望 (4,5), 得到 {dx.shape}"
    assert 'W' in layer.grads
    assert 'b' in layer.grads
    print("  ✓ Dense 前向/反向传播")


def test_activation_forward_backward():
    """测试激活函数"""
    x = np.array([[-2, -1, 0, 1, 2]], dtype=float)
    grad = np.ones_like(x)

    # Sigmoid
    s = Sigmoid()
    out = s.forward(x)
    dx = s.backward(grad)
    assert out.shape == x.shape
    assert np.all(out > 0) and np.all(out < 1)
    print("  ✓ Sigmoid")

    # Tanh
    t = Tanh()
    out = t.forward(x)
    dx = t.backward(grad)
    assert out.shape == x.shape
    assert np.all(np.abs(out) <= 1)
    print("  ✓ Tanh")

    # ReLU
    r = ReLU()
    out = r.forward(x)
    dx = r.backward(grad)
    assert out.shape == x.shape
    assert np.all(out[0, :3] == 0)  # 负数部分为 0
    print("  ✓ ReLU")

    # Softmax
    sm = Softmax()
    out = sm.forward(x)
    assert np.allclose(np.sum(out, axis=-1), 1.0)
    print("  ✓ Softmax")

    # LeakyReLU
    lr = LeakyReLU(0.1)
    out = lr.forward(x)
    dx = lr.backward(grad)
    assert out.shape == x.shape
    print("  ✓ LeakyReLU")


def test_loss_functions():
    """测试损失函数"""
    y_pred = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    y_true = np.array([0, 1])

    # MSE
    mse = MSELoss()
    loss = mse.forward(y_pred, to_one_hot(y_true, 2))
    grad = mse.backward()
    assert grad.shape == y_pred.shape
    print("  ✓ MSELoss")

    # CrossEntropy
    ce = CrossEntropyLoss()
    logits = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=float)
    loss = ce.forward(logits, y_true)
    grad = ce.backward()
    assert np.sum(grad[0]) < 1e-10  # Softmax + CE 梯度之和为 0
    assert grad.shape == logits.shape
    print("  ✓ CrossEntropyLoss")

    # BinaryCrossEntropy (接收 logits)
    bce = BinaryCrossEntropyLoss()
    y_pred_logits = np.array([[2.0], [-1.0]], dtype=float)  # logits
    y_true_binary = np.array([[1], [0]], dtype=float)
    loss = bce.forward(y_pred_logits, y_true_binary)
    grad = bce.backward()
    assert grad.shape == y_pred_logits.shape
    print("  ✓ BinaryCrossEntropyLoss")


def test_optimizer_step():
    """测试优化器"""
    params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
    grads = {'W': np.array([[0.1, 0.1], [0.1, 0.1]])}

    # SGD
    sgd = SGD(lr=0.5)
    sgd.step(params, grads)
    assert np.allclose(params['W'], [[0.95, 1.95], [2.95, 3.95]])
    print("  ✓ SGD")

    # Adam
    params_adam = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
    adam = Adam(lr=0.1)
    old_val = params_adam['W'].copy()
    adam.step(params_adam, grads)
    assert params_adam['W'].shape == old_val.shape
    assert not np.allclose(params_adam['W'], old_val)  # 参数更新了
    print("  ✓ Adam")


def test_conv2d_forward_backward():
    """测试卷积层"""
    layer = Conv2D(out_channels=2, kernel_size=3, padding=1)
    x = np.random.randn(2, 1, 5, 5)
    out = layer.forward(x)
    assert out.shape == (2, 2, 5, 5), f"期望 (2,2,5,5), 得到 {out.shape}"

    grad = np.random.randn(2, 2, 5, 5)
    dx = layer.backward(grad)
    assert dx.shape == (2, 1, 5, 5), f"期望 (2,1,5,5), 得到 {dx.shape}"
    print("  ✓ Conv2D 前向/反向传播")


def test_pooling():
    """测试池化层"""
    x = np.random.randn(2, 3, 4, 4)

    # MaxPool
    maxpool = MaxPool2D(pool_size=2, stride=2)
    out = maxpool.forward(x)
    assert out.shape == (2, 3, 2, 2)
    dx = maxpool.backward(np.ones_like(out))
    assert dx.shape == x.shape
    print("  ✓ MaxPool2D")

    # AvgPool
    avgpool = AvgPool2D(pool_size=2, stride=2)
    out = avgpool.forward(x)
    assert out.shape == (2, 3, 2, 2)
    dx = avgpool.backward(np.ones_like(out))
    assert dx.shape == x.shape
    print("  ✓ AvgPool2D")


def test_flatten():
    """测试 Flatten 层"""
    layer = Flatten()
    x = np.random.randn(4, 3, 5, 5)
    out = layer.forward(x)
    assert out.shape == (4, 75)

    grad = np.random.randn(4, 75)
    dx = layer.backward(grad)
    assert dx.shape == (4, 3, 5, 5)
    print("  ✓ Flatten")


def test_dropout():
    """测试 Dropout 层"""
    layer = Dropout(rate=0.5)
    x = np.ones((100, 100))

    # 训练模式
    layer.train()
    out = layer.forward(x)
    assert out.shape == x.shape
    # 大约一半被丢弃 (乘以 keep_prob 缩放)
    zeros_ratio = np.mean(out == 0)
    assert 0.3 < zeros_ratio < 0.7, f"丢弃比例异常: {zeros_ratio}"
    print("  ✓ Dropout (训练模式)")

    # 评估模式
    layer.eval()
    out = layer.forward(x)
    assert np.allclose(out, x)
    print("  ✓ Dropout (评估模式)")


def test_embedding():
    """测试 Embedding 层"""
    layer = Embedding(vocab_size=10, embedding_dim=8)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    out = layer.forward(x)
    assert out.shape == (2, 3, 8)

    grad = np.random.randn(2, 3, 8)
    dx = layer.backward(grad)
    assert layer.grads['W'].shape == (10, 8)
    print("  ✓ Embedding")


def test_attention():
    """测试注意力机制"""
    batch_size, seq_len, d_model = 2, 6, 16

    # SelfAttention
    sa = SelfAttention(embed_dim=d_model)
    x = np.random.randn(batch_size, seq_len, d_model)
    out = sa.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)

    grad = np.random.randn(batch_size, seq_len, d_model)
    dx = sa.backward(grad)
    assert dx.shape == (batch_size, seq_len, d_model)
    print("  ✓ SelfAttention")

    # MultiHeadAttention
    mha = MultiHeadAttention(embed_dim=d_model, num_heads=4)
    out = mha.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)

    dx = mha.backward(grad)
    assert dx.shape == (batch_size, seq_len, d_model)
    print("  ✓ MultiHeadAttention")


def test_transformer_block():
    """测试 TransformerBlock"""
    batch_size, seq_len, d_model = 2, 8, 32
    block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=64)

    x = np.random.randn(batch_size, seq_len, d_model)
    out = block.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)

    grad = np.random.randn(batch_size, seq_len, d_model)
    dx = block.backward(grad)
    assert dx.shape == (batch_size, seq_len, d_model)
    print("  ✓ TransformerBlock")


def test_positional_encoding():
    """测试位置编码"""
    pe = PositionalEncoding(max_len=10, d_model=16)
    x = np.random.randn(2, 8, 16)
    out = pe.forward(x)
    assert out.shape == x.shape
    # 检查是否添加了位置信息
    assert not np.allclose(out, x)
    print("  ✓ PositionalEncoding")


def test_sequential_forward_backward():
    """测试 Sequential 容器"""
    network = Sequential([
        Dense(units=8),
        ReLU(),
        Dense(units=1),
    ])

    x = np.random.randn(4, 5)
    out = network.forward(x)
    assert out.shape == (4, 1)  # logits output

    grad = np.ones_like(out)
    dx = network.backward(grad)
    assert dx.shape == (4, 5)

    assert network.num_params() > 0
    print("  ✓ Sequential 前向/反向传播")


def test_model_training():
    """测试完整的模型训练流程"""
    # 生成简单的线性数据
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

    # 构建小网络 (输出 logits, Sigmoid 由 BinaryCrossEntropyLoss 内部处理)
    network = Sequential([
        Dense(units=4),
        Tanh(),
        Dense(units=1),
    ])

    model = Model(network)
    model.compile(
        loss=BinaryCrossEntropyLoss(),
        optimizer=SGD(lr=0.1)
    )

    history = model.fit(X, y, epochs=5, batch_size=16, verbose=False)
    assert len(history['loss']) == 5
    print("  ✓ 模型训练流程")


def test_normalization():
    """测试归一化层"""
    x = np.random.randn(4, 3, 8, 8)

    # BatchNorm
    bn = BatchNormalization()
    bn.train()
    out = bn.forward(x)
    assert out.shape == x.shape

    grad = np.random.randn(*x.shape)
    dx = bn.backward(grad)
    assert dx.shape == x.shape
    print("  ✓ BatchNormalization")


def test_utils():
    """测试工具函数"""
    # make_moons
    X, y = make_moons(n_samples=50, noise=0.05, seed=42)
    assert X.shape == (50, 2)
    assert y.shape == (50,)

    # to_one_hot
    one_hot = to_one_hot(y.astype(int), 2)
    assert one_hot.shape == (50, 2)

    # train_test_split
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_size=0.2, seed=42)
    assert X_tr.shape[0] == 40
    assert X_te.shape[0] == 10

    # batch_iterator
    batches = list(batch_iterator(X_tr, y_tr, batch_size=8))
    assert len(batches) == 5

    # accuracy
    acc = accuracy(np.array([0, 1, 0]), np.array([0, 1, 1]))
    assert acc == 2/3

    print("  ✓ 工具函数")


if __name__ == '__main__':
    print("=" * 60)
    print("运行所有单元测试")
    print("=" * 60)

    tests = [
        ("Dense", test_dense_forward_backward),
        ("Activations", test_activation_forward_backward),
        ("Loss Functions", test_loss_functions),
        ("Optimizers", test_optimizer_step),
        ("Conv2D", test_conv2d_forward_backward),
        ("Pooling", test_pooling),
        ("Flatten", test_flatten),
        ("Dropout", test_dropout),
        ("Embedding", test_embedding),
        ("Attention", test_attention),
        ("Transformer", test_transformer_block),
        ("Positional Encoding", test_positional_encoding),
        ("Sequential", test_sequential_forward_backward),
        ("Normalization", test_normalization),
        ("Model Training", test_model_training),
        ("Utils", test_utils),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败")
    print(f"{'=' * 60}")
