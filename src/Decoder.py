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


class MultiHeadCrossAttention():

    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = Linear(d_model, 2 * d_model) 
        self.q_layer = Linear(d_model, d_model)
        self.linear_layer = Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.shape 
        print(f"x.shape: {x.shape}")
        kv = self.kv_layer(x) 
        print(f"kv.shape: {kv.shape}")
        q = self.q_layer(y) 
        print(f"q.shape: {q.shape}")
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim) 
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) 
        q = np.transpose(q, (0, 2, 1, 3)) 
        k, v = np.split(kv, 2, axis=-1) 
        values, attention = scaled_dot_product_attention(q, k, v, mask) 
        print(f"values.shape: {values.shape}, attention.shape: {attention.shape}")
        values = values.reshape(batch_size, sequence_length, d_model) 
        out = self.linear_layer(values)  
        print(f"out.shape after passing through linear layer: {out.shape}")
        return out  

class DecoderLayer():

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        self.self_attention = MultiHeadAttention(input_dim=d_model, d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden_units=ffn_hidden, drop_prop=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y 
        print("MASKED SELF ATTENTION")
        y = self.self_attention.forward(y, mask=decoder_mask) 
        print("DROP OUT 1")
        y = self.dropout1(y) 
        print("ADD + LAYER NORMALIZATION 1")
        y = self.norm1.forward(y + _y) 

        _y = y 
        print("CROSS ATTENTION")
        y = self.encoder_decoder_attention.forward(x, y, mask=None) 
        print("DROP OUT 2")  
        y = self.dropout2(y)
        print("ADD + LAYER NORMALIZATION 2")
        y = self.norm2.forward(y + _y)  

        _y = y  
        print("FEED FORWARD 1")
        y = self.ffn.forward(y) 
        print("DROP OUT 3")
        y = self.dropout3(y) 
        print("ADD + LAYER NORMALIZATION 3")
        y = self.norm3.forward(y + _y) 
        return y 

class SequentialDecoder():
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, *inputs):
        x, y, mask = inputs
        for layer in self.layers:
            y = layer.forward(x, y, mask) 
        return y

class Decoder():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self, x, y, mask):
        y = self.layers.forward(x, y, mask)
        return y 