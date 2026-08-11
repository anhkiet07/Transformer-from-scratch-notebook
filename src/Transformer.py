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
    def __init__(self, d_model, max_sequence_length=5000):
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        even_i = np.arange(0, self.d_model, 2).astype(np.float32)
        even_dominator = np.power(10000, even_i / self.d_model)

        odd_i = np.arange(1, self.d_model, 2).astype(np.float32)
        odd_dominator = np.power(10000, (odd_i - 1) / self.d_model)

        denominator = even_dominator

        position = np.arange(self.max_sequence_length, dtype = np.float32).reshape(self.max_sequence_length, 1)

        even_PE = np.sin(position / denominator)
        odd_PE = np.cos(position / denominator)

        stacked = np.stack((even_PE, odd_PE), axis = 2)
        PE = stacked.reshape(stacked.shape[0], -1)

        return PE

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) * 0.01

    def __call__(self, x):
        return self.weight[x]

class SentenceEmbedding:
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return np.array(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = np.stack(tokenized)
        return tokenized

    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder.forward()
        x = self.dropout(x + pos)
        return x

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
class Dropout:
    def __init__(self, p):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * mask / (1 - self.p)

class EncoderLayer():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prop):
        self.attention = MultiHeadAttention(input_dim = d_model, d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = Dropout(p = drop_prop)
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden_units = ffn_hidden, drop_prop = drop_prop)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = Dropout(p = drop_prop)

    def forward(self, x, self_attention_mask):
        residual_x = x
        x = self.attention.forward(x, mask = self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1.forward(x + residual_x)
        residual_x = x
        x = self.ffn.forward(x)
        x = self.dropout2(x)
        x = self.norm2.forward(x + residual_x)
        return x
class SequentialEncoder:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, self_attention_mask):
        for layer in self.layers:
            x = layer.forward(x, self_attention_mask)
        return x
class Encoder():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                           for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding.forward(x, start_token, end_token)
        x = self.layers.forward(x, self_attention_mask)
        return x

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
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        q = np.transpose(q, (0, 2, 1, 3))
        kv = np.transpose(kv, (0, 2, 1, 3))
        k, v = np.split(kv, 2, axis=-1)
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = np.transpose(values, (0, 2, 1, 3)).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
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

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y
        y = self.self_attention.forward(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1.forward(y + _y)

        _y = y
        y = self.encoder_decoder_attention.forward(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2.forward(y + _y)

        _y = y
        y = self.ffn.forward(y)
        y = self.dropout3(y)
        y = self.norm3.forward(y + _y)
        return y

class SequentialDecoder():
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for layer in self.layers:
            y = layer.forward(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder():
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding.forward(y, start_token, end_token)
        y = self.layers.forward(x, y, self_attention_mask, cross_attention_mask)
        return y

class Transformer():
    def __init__(self,
                d_model,
                ffn_hidden,
                num_heads,
                drop_prob,
                num_layers,
                max_sequence_length,
                kn_vocab_size,
                english_to_index,
                kannada_to_index,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN
                ):
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = Linear(d_model, kn_vocab_size)

    def forward(self,
                x,
                y,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder.forward(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder.forward(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out
