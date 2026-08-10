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
    attention = softmax(scaled, axis=-1)
    out = np.matmul(attention, v)
    return out, attention


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Linear:
    def __init__(self, in_features, out_features):
        self.W = Parameter(np.random.randn(in_features, out_features) / np.sqrt(in_features))
        self.b = Parameter(np.zeros((out_features,)))

    def __call__(self, x):
        return np.matmul(x, self.W.data) + self.b.data


class MultiHeadAttention:
    def __init__(self, input_dim, d_model, num_heads):
        self.input_dim = input_dim
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads #64
        self.qkv_linear = Linear(input_dim, d_model * 3) #512 x 1536
        self.out_linear = Linear(d_model, d_model)


    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.shape
        print(f"x.shape: {x.shape}")
        qkv = self.qkv_linear(x)
        print(f"qkv.shape: {qkv.shape}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.shape after reshape: {qkv.shape}")
        qkv = np.transpose(qkv, (0, 2, 1, 3))
        print(f"qkv.shape after transpose: {qkv.shape}")
        q, k, v = np.split(qkv, 3, axis=-1)
        print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        print(f"values.shape: {values.shape}, attention.shape: {attention.shape}")
        values = values.reshape(batch_size, sequence_length, self.d_model)
        print(f"values.shape after reshape: {values.shape}")
        out = self.out_linear(values)
        return out
    
class Dropout:
    def __init__(self, p):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * mask / (1 - self.p)


class LayerNormalization:
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = np.ones(parameters_shape)
        self.beta = np.zeros(parameters_shape)

    def forward(self, inputs):
        dims = tuple(-(i + 1) for i in range(len(self.parameters_shape)))
        mean = inputs.mean(axis=dims, keepdims=True)
        var = ((inputs - mean) ** 2).mean(axis=dims, keepdims=True)
        std = np.sqrt(var + self.eps)
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward:
    def __init__(self, d_model, hidden_units, drop_prop):
        self.linear1 = Linear(d_model, hidden_units)
        self.linear2 = Linear(hidden_units, d_model)
        self.drop_prop = drop_prop
        self.training = True

    def relu(self, x):
        return np.maximum(0, x)

    def dropout(self, x):
        if not self.training or self.drop_prop == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.drop_prop).astype(x.dtype)
        return x * mask / (1 - self.drop_prop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prop):
        self.attention = MultiHeadAttention(input_dim = d_model, d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = Dropout(p = drop_prop)
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden_units = ffn_hidden, drop_prop = drop_prop)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = Dropout(p = drop_prop)

    def forward(self, x):
        residual_x = x
        x = self.attention.forward(x, mask = None)
        x = self.dropout1(x)
        x = self.norm1.forward(x + residual_x)
        residual_x = x
        x = self.ffn.forward(x)
        x = self.dropout2(x)
        x = self.norm2.forward(x + residual_x)
        return x

class Encoder():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        self.layers = [EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                        for _ in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
