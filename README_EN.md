# DL From Scratch

> A pure NumPy implementation of machine learning & deep learning algorithms — **every forward pass, backward pass, and mathematical derivation is manually implemented from scratch**.

## Project Goals

- ✅ **No PyTorch / TensorFlow / sklearn** — Only NumPy
- ✅ **Modular Design** — Fully decoupled components
- ✅ **Education-Friendly** — Clean, well-commented code
- ✅ **Progressive Learning** — From traditional ML to modern Transformer

## Project Structure

```
dl_from_scratch/
├── ml/             # Traditional ML algorithms
│   ├── linear_regression.py   # Linear Regression (OLS + GD)
│   ├── logistic_regression.py # Logistic Regression
│   ├── knn.py                 # K-Nearest Neighbors
│   ├── kmeans.py              # K-Means Clustering
│   ├── decision_tree.py       # Decision Tree (CART)
│   ├── naive_bayes.py         # Gaussian Naive Bayes
│   ├── pca.py                 # Principal Component Analysis
│   └── svm.py                 # Support Vector Machine (SMO)
├── core/           # Foundational components
│   └── initializers.py   # Weight initializers (He, Xavier, etc.)
├── layers/         # Deep learning layers
│   ├── base.py           # Layer base class
│   ├── dense.py          # Fully connected layer
│   ├── convolution.py    # Conv2D (im2col implementation)
│   ├── pooling.py        # MaxPool2D, AvgPool2D
│   ├── flatten.py        # Flatten layer
│   ├── dropout.py        # Dropout layer
│   ├── normalization.py  # Batch/Layer Normalization
│   ├── embedding.py      # Token Embedding
│   ├── attention.py      # Self-Attention & Multi-Head Attention
│   ├── transformer.py    # TransformerBlock & PositionalEncoding
│   └── sequential.py     # Sequential container
├── activations/    # Activation functions
│   └── functions.py      # ReLU, Sigmoid, Tanh, Softmax, GELU, etc.
├── losses/         # Loss functions
│   └── functions.py      # MSE, CrossEntropy, BCE, L1
├── optimizers/     # Optimizers
│   └── sgd.py            # SGD, Momentum, Adam, RMSprop
├── models/         # Model wrapper
│   └── model.py          # train/predict/evaluate API
└── utils/          # Utilities
    ├── data.py           # Data generation
    └── metrics.py        # Evaluation metrics
```

## Examples

### Machine Learning

| Example | Content | Keywords |
|---------|---------|----------|
| `06_linear_regression.py` | Linear Regression — House Price Prediction | Normal Equation, Gradient Descent, R² |
| `07_knn.py` | K-Nearest Neighbors — Classification | Lazy Learning, Distance Metrics, K Value |
| `08_kmeans.py` | K-Means Clustering | Unsupervised Learning, K-Means++, Elbow Method |
| `09_decision_tree.py` | Decision Tree — Classification | Gini Impurity, Recursive Splitting, Pruning |
| `10_pca.py` | PCA Dimensionality Reduction | Eigendecomposition, Explained Variance |

### Deep Learning

| Example | Content | Keywords |
|---------|---------|----------|
| `01_single_neuron.py` | Single Neuron — AND Gate | Single-Layer NN, Sigmoid, BCE |
| `02_multi_layer_perceptron.py` | MLP — Moon Classification | Fully Connected, Backpropagation, Decision Boundary |
| `03_cnn_mnist.py` | CNN — MNIST Digit Recognition | Convolution, Pooling, Image Classification |
| `04_transformer_demo.py` | Transformer — Sequence Modeling | Self-Attention, Multi-Head Attention, Positional Encoding |
| `05_attention_visualization.py` | Attention Visualization | Attention Weights, QKV, Causal Mask |

## Quick Start

```bash
pip install numpy matplotlib
python examples/06_linear_regression.py      # Start with traditional ML
python examples/01_single_neuron.py          # Or start with deep learning
```

> **Note**: `scikit-learn` is only used by `03_cnn_mnist.py` for loading the MNIST dataset. The core implementation does not depend on it.
> To run the CNN example: `pip install scikit-learn`

## Algorithm List

| Algorithm | Type | File |
|-----------|------|------|
| Linear Regression | Regression (Closed-form + GD) | `ml/linear_regression.py` |
| Logistic Regression | Binary Classification | `ml/logistic_regression.py` |
| K-Nearest Neighbors | Classification / Regression | `ml/knn.py` |
| K-Means | Clustering (K-Means++) | `ml/kmeans.py` |
| Decision Tree | Classification / Regression (CART) | `ml/decision_tree.py` |
| Gaussian Naive Bayes | Classification | `ml/naive_bayes.py` |
| PCA | Dimensionality Reduction | `ml/pca.py` |
| SVM (SMO) | Binary Classification | `ml/svm.py` |
| Dense (Fully Connected) | Deep Learning | `layers/dense.py` |
| Conv2D | Deep Learning | `layers/convolution.py` |
| Self-Attention | Deep Learning | `layers/attention.py` |
| Transformer | Deep Learning | `layers/transformer.py` |

## License

This project is for **educational purposes**. Feel free to use, modify, and share.
