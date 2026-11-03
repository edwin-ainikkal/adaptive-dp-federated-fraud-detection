"""
common/model.py
----------------
A small 2-hidden-layer Multi-Layer Perceptron for binary credit-card-fraud
classification, implemented in plain NumPy.

Why NumPy instead of PyTorch/TensorFlow?
  - This project runs across TWO DIFFERENT DESKTOPS. The single biggest source
    of "it works on my machine but not the other one" pain in student projects
    is deep-learning-framework / CUDA driver mismatches between two computers.
  - A 30-feature tabular MLP is tiny. NumPy forward/backward passes are fast
    enough, fully deterministic, and install identically everywhere with
    `pip install numpy` -- no GPU, no CUDA, no version roulette.
  - It also makes every line of the adaptive-DP math (clipping, per-layer
    noise, Fisher-style importance) fully transparent for your viva/defense,
    since nothing is hidden inside a framework's autograd engine.
"""

import numpy as np

LAYER_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]


class FraudMLP:
    def __init__(self, input_dim=30, h1=24, h2=12, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, h1)).astype(np.float32),
            "b1": np.zeros(h1, dtype=np.float32),
            "W2": rng.normal(0, np.sqrt(2.0 / h1), size=(h1, h2)).astype(np.float32),
            "b2": np.zeros(h2, dtype=np.float32),
            "W3": rng.normal(0, np.sqrt(2.0 / h2), size=(h2, 1)).astype(np.float32),
            "b3": np.zeros(1, dtype=np.float32),
        }

    def forward(self, X):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.params["W3"] + self.params["b3"]
        cache = (X, z1, a1, z2, a2, z3)
        return z3.squeeze(-1), cache

    def backward(self, cache, y, pos_weight=1.0):
        """Binary cross-entropy gradient (logits form), with an optional
        positive-class weight to help with the ~0.17% fraud class imbalance."""
        X, z1, a1, z2, a2, z3 = cache
        n = X.shape[0]
        logits = z3.squeeze(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        sample_w = np.where(y == 1, pos_weight, 1.0)

        dz3 = ((p - y) * sample_w / n).reshape(-1, 1)
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.params["W3"].T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        loss = -np.mean(sample_w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        return grads, float(loss)

    def per_sample_backward(self, cache, y, pos_weight=1.0):
        """Backward pass that keeps PER-SAMPLE quantities (needed for DP-SGD).
        For a linear layer, one sample's gradient is the outer product
        input_i (x) dz_i, so its Frobenius norm is simply ||input_i|| * ||dz_i||
        -- per-sample norms are obtained without materialising per-sample
        gradient tensors. Returns ((inputs, dzs), mean_loss) where
        inputs=[X, a1, a2] and dzs=[dz1, dz2, dz3] (one row per sample,
        NOT divided by the batch size)."""
        X, z1, a1, z2, a2, z3 = cache
        logits = z3.squeeze(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        w = np.where(y == 1, pos_weight, 1.0)
        dz3 = ((p - y) * w).reshape(-1, 1)
        dz2 = (dz3 @ self.params["W3"].T) * (z2 > 0)
        dz1 = (dz2 @ self.params["W2"].T) * (z1 > 0)
        loss = -np.mean(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        return ([X, a1, a2], [dz1, dz2, dz3]), float(loss)

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def get_state(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_state(self, state):
        for k in LAYER_NAMES:
            self.params[k] = np.array(state[k], dtype=np.float32).copy()


class AdamOptimizer:
    """A minimal per-parameter Adam optimizer (reset fresh at the start of
    each client's local-training phase, every FL round)."""

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
