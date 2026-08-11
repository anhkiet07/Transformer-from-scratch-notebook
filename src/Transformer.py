import numpy as np
import math

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled, axis = -1)
    out = np.matmul(attention, v)
    return out, attention

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_sequence_length = self.max_sequence_length

    def forward(self):
        even_i = np.arange(0, self.d_model, 2).astype(np.float32)
        even_dominator = np.power(10000, even_i / self.d_model)

        odd_i = np.arange(1, self.d_model, 2).astype(np.float32)
        odd_dominator = np.power(10000, (odd_i - 1) / self.d_model)

        denominator = even_dominator

        position = np.arrange(self.max_sequence_length, dtype = np.float32).reshape(self.max_sequence_length, 1)

        even_PE = np.sin(position / denominator)
        odd_PE = np.cos(position / denominator)

        stacked = np.stack((even_PE, odd_PE), axis = 2)
        PE = stacked.reshape(stacked.shape[0], -1)

        return PE

