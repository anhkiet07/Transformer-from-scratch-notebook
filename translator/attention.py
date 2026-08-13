"""
Multi-head (self- and cross-) attention, with backward passes.

Note for readers of docs/known-issues.md: that document flags two bugs in
the original src/ NumPy code -- a missing transpose before the final
reshape in MultiHeadAttention, and a missing transpose of `kv` in
Decoder.py's MultiHeadCrossAttention. Both are fixed here (see the
`np.transpose(out, (0, 2, 1, 3))` calls below) -- required for the forward
pass to even be numerically correct, which matters once gradients depend on
it.
"""

import numpy as np

from translator.layers import Linear, sdpa_backward, sdpa_forward


class MultiHeadAttention:
    """Self-attention: Q, K, V all come from the same input x."""

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_linear = Linear(d_model, d_model * 3)
        self.out_linear = Linear(d_model, d_model)
        self._cache = None

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        qkv = self.qkv_linear.forward(x)
        qkv = qkv.reshape(B, S, self.num_heads, 3 * self.head_dim)
        qkv = np.transpose(qkv, (0, 2, 1, 3))
        q, k, v = np.split(qkv, 3, axis=-1)
        out, _, sdpa_cache = sdpa_forward(q, k, v, mask)
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, S, self.d_model)
        result = self.out_linear.forward(out)
        self._cache = (B, S, sdpa_cache)
        return result

    def backward(self, dout):
        B, S, sdpa_cache = self._cache
        d_merged = self.out_linear.backward(dout)
        d_out = d_merged.reshape(B, S, self.num_heads, self.head_dim)
        d_out = np.transpose(d_out, (0, 2, 1, 3))
        dq, dk, dv = sdpa_backward(d_out, sdpa_cache)
        dqkv = np.concatenate([dq, dk, dv], axis=-1)
        dqkv = np.transpose(dqkv, (0, 2, 1, 3)).reshape(B, S, 3 * self.d_model)
        return self.qkv_linear.backward(dqkv)

    def parameters(self):
        return self.qkv_linear.parameters() + self.out_linear.parameters()


class MultiHeadCrossAttention:
    """Cross-attention: K, V come from x (encoder output), Q comes from y (decoder state)."""

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_linear = Linear(d_model, d_model * 2)
        self.q_linear = Linear(d_model, d_model)
        self.out_linear = Linear(d_model, d_model)
        self._cache = None

    def forward(self, x, y, mask=None):
        B, S, _ = x.shape
        kv = self.kv_linear.forward(x)
        q = self.q_linear.forward(y)
        kv = kv.reshape(B, S, self.num_heads, 2 * self.head_dim)
        kv = np.transpose(kv, (0, 2, 1, 3))
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        q = np.transpose(q, (0, 2, 1, 3))
        k, v = np.split(kv, 2, axis=-1)
        out, _, sdpa_cache = sdpa_forward(q, k, v, mask)
        out = np.transpose(out, (0, 2, 1, 3)).reshape(B, S, self.d_model)
        result = self.out_linear.forward(out)
        self._cache = (B, S, sdpa_cache)
        return result

    def backward(self, dout):
        """Returns (dx, dy) -- gradients w.r.t. the encoder output and the decoder state."""
        B, S, sdpa_cache = self._cache
        d_merged = self.out_linear.backward(dout)
        d_out = d_merged.reshape(B, S, self.num_heads, self.head_dim)
        d_out = np.transpose(d_out, (0, 2, 1, 3))
        dq, dk, dv = sdpa_backward(d_out, sdpa_cache)

        dkv = np.concatenate([dk, dv], axis=-1)
        dkv = np.transpose(dkv, (0, 2, 1, 3)).reshape(B, S, 2 * self.d_model)
        dx = self.kv_linear.backward(dkv)

        dq = np.transpose(dq, (0, 2, 1, 3)).reshape(B, S, self.d_model)
        dy = self.q_linear.backward(dq)

        return dx, dy

    def parameters(self):
        return self.kv_linear.parameters() + self.q_linear.parameters() + self.out_linear.parameters()
