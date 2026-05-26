"""模型封装 — 训练/预测/评估接口"""

import numpy as np


class Model:
    """深度学习模型封装

    封装 Sequential 或任意 Layer，提供 compile / fit / predict / evaluate 接口。

    Args:
        network: Layer 或 Sequential 实例
    """

    def __init__(self, network):
        self.network = network
        self.loss_fn = None
        self.optimizer = None
        self.history = {'loss': [], 'val_loss': []}
        self._compiled = False

    def compile(self, loss, optimizer):
        """配置模型

        Args:
            loss: 损失函数实例
            optimizer: 优化器实例
        """
        self.loss_fn = loss
        self.optimizer = optimizer
        self._compiled = True

    def forward(self, x):
        """前向传播"""
        return self.network.forward(x)

    def backward(self, grad):
        """反向传播"""
        return self.network.backward(grad)

    def fit(self, X, y, epochs=10, batch_size=32, verbose=True,
            X_val=None, y_val=None, shuffle=True):
        """训练模型

        Args:
            X: 训练数据, shape=(n_samples, ...)
            y: 训练标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印日志
            X_val: 验证数据 (可选)
            y_val: 验证标签 (可选)
            shuffle: 是否打乱数据

        Returns:
            训练历史 { 'loss': [...], 'val_loss': [...] }
        """
        if not self._compiled:
            raise RuntimeError("模型未编译，请先调用 compile()")

        n_samples = X.shape[0]
        self.history = {'loss': []}
        if X_val is not None:
            self.history['val_loss'] = []

        for epoch in range(1, epochs + 1):
            # 打乱数据
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            epoch_loss = 0.0
            num_batches = 0

            # 批次训练
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                loss = self._train_on_batch(X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.history['loss'].append(avg_loss)

            # 验证
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)

            # 打印日志
            if verbose:
                log = f"Epoch {epoch:3d}/{epochs} | loss: {avg_loss:.6f}"
                if val_loss is not None:
                    log += f" | val_loss: {val_loss:.6f}"
                print(log)

        return self.history

    def _train_on_batch(self, X_batch, y_batch):
        """训练单个批次"""
        # 前向传播
        y_pred = self.network.forward(X_batch)

        # 计算损失
        loss = self.loss_fn.forward(y_pred, y_batch)

        # 反向传播
        grad = self.loss_fn.backward()
        self.network.backward(grad)

        # 更新参数
        params = self.network.get_params()
        grads = self.network.get_grads()
        self.optimizer.step(params, grads)

        return loss

    def predict(self, X):
        """预测

        Args:
            X: 输入数据, shape=(n_samples, ...)

        Returns:
            预测结果
        """
        self.network.eval()
        return self.network.forward(X)

    def evaluate(self, X, y):
        """评估模型

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            损失值
        """
        self.network.eval()
        y_pred = self.network.forward(X)
        loss = self.loss_fn.forward(y_pred, y)
        return float(loss)

    def summary(self):
        """打印模型结构"""
        print("=" * 60)
        print(f"{'Layer':<30} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 60)

        total_params = 0
        # 获取网络结构
        if hasattr(self.network, 'layers'):
            layers = self.network.layers
        else:
            layers = [self.network]

        x_shape = None
        for i, layer in enumerate(layers):
            n_params = layer.num_params()
            total_params += n_params
            print(f"{i}: {str(layer):<28} {str(x_shape):<20} {n_params:<10}")

        print("=" * 60)
        print(f"Total params: {total_params}")
        print("=" * 60)

    def get_params(self):
        return self.network.get_params()

    def get_grads(self):
        return self.network.get_grads()

    def zero_grad(self):
        self.network.zero_grad()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()
