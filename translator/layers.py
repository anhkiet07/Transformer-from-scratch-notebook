"""
Core NumPy building blocks, each with a forward() that caches what it needs
and a backward() that returns the gradient w.r.t. its input(s) while
accumulating gradients into any Parameter it owns.

This is a hand-written backprop implementation (no autograd library). Every
formula here is standard textbook calculus for these ops; see
translator/test_gradients.py for a finite-difference check that verifies
each one against numerical gradients.
"""

import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)


class Linear:
    def __init__(self, in_features, out_features):
        limit = 1.0 / np.sqrt(in_features)
        self.W = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W.data + self.b.data

    def backward(self, dout):
        x_flat = self._x.reshape(-1, self._x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        self.W.grad += x_flat.T @ dout_flat
        self.b.grad += dout_flat.sum(axis=0)
        return dout @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = Parameter(np.random.randn(vocab_size, d_model) * 0.01)
        self._idx = None

    def forward(self, idx):
        self._idx = idx
        return self.weight.data[idx]

    def backward(self, dout):
        np.add.at(self.weight.grad, self._idx, dout)
        return None

    def parameters(self):
        return [self.weight]


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Dropout:
    def __init__(self, p):
        self.p = p
        self.training = True
        self._mask = None

    def forward(self, x):
        if not self.training or self.p == 0:
            self._mask = None
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * self._mask / (1 - self.p)

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask / (1 - self.p)


class LayerNormalization:
    """Normalizes over the last axis. eps is fixed (not learned)."""

    def __init__(self, d_model, eps=1e-5):
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        xhat = (x - mu) / std
        out = self.gamma.data * xhat + self.beta.data
        self._cache = (x, mu, std, xhat)
        return out

    def backward(self, dout):
        x, mu, std, xhat = self._cache
        D = x.shape[-1]
        reduce_axes = tuple(range(dout.ndim - 1))
        self.gamma.grad += np.sum(dout * xhat, axis=reduce_axes)
        self.beta.grad += np.sum(dout, axis=reduce_axes)

        dxhat = dout * self.gamma.data
        dvar = np.sum(dxhat * (x - mu) * -0.5 * std ** -3, axis=-1, keepdims=True)
        dmu = np.sum(dxhat * (-1.0 / std), axis=-1, keepdims=True) \
            + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True)
        dx = dxhat / std + dvar * 2.0 * (x - mu) / D + dmu / D
        return dx

    def parameters(self):
        return [self.gamma, self.beta]


class PositionalEncoding:
    """Fixed sinusoidal encoding -- no parameters, no gradient."""

    def __init__(self, d_model, max_sequence_length):
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        even_i = np.arange(0, d_model, 2, dtype=np.float32)
        denominator = np.power(10000, even_i / d_model)
        position = np.arange(max_sequence_length, dtype=np.float32).reshape(-1, 1)
        even_pe = np.sin(position / denominator)
        odd_pe = np.cos(position / denominator)
        stacked = np.stack((even_pe, odd_pe), axis=2)
        self.pe = stacked.reshape(stacked.shape[0], -1)

    def forward(self):
        return self.pe


def softmax_forward(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(dout, softmax_out, axis=-1):
    dot = np.sum(dout * softmax_out, axis=axis, keepdims=True)
    return softmax_out * (dout - dot)


def sdpa_forward(q, k, v, mask=None):
    """Scaled dot-product attention. q,k,v: (..., S, d_k)."""
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = softmax_forward(scores, axis=-1)
    out = np.matmul(attn, v)
    cache = (q, k, v, attn, d_k)
    return out, attn, cache


def sdpa_backward(dout, cache):
    """Returns (dq, dk, dv) given dout = dL/d(out)."""
    q, k, v, attn, d_k = cache
    d_attn = np.matmul(dout, np.swapaxes(v, -2, -1))
    dv = np.matmul(np.swapaxes(attn, -2, -1), dout)
    d_raw = softmax_backward(d_attn, attn, axis=-1) / np.sqrt(d_k)
    dq = np.matmul(d_raw, k)
    dk = np.matmul(np.swapaxes(d_raw, -2, -1), q)
    return dq, dk, dv
