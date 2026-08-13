"""
Assembles the layers in translator/layers.py and translator/attention.py
into EncoderLayer/DecoderLayer/Encoder/Decoder/Transformer, each with a
matching backward() -- plus the cross-entropy loss used to train them.

Architecture mirrors src/Transformer.py (character-level tokenization,
sinusoidal positions, post-norm residual blocks), but every module here
also implements backward() and parameters(), which src/ intentionally does
not (see docs/known-issues.md #4).
"""

import numpy as np

from translator.attention import MultiHeadAttention, MultiHeadCrossAttention
from translator.layers import Dropout, Embedding, Linear, LayerNormalization, PositionalEncoding, ReLU, softmax_forward


class PositionwiseFeedForward:
    def __init__(self, d_model, hidden_units, drop_prob):
        self.linear1 = Linear(d_model, hidden_units)
        self.relu = ReLU()
        self.dropout = Dropout(drop_prob)
        self.linear2 = Linear(hidden_units, d_model)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.dropout.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, dout):
        d = self.linear2.backward(dout)
        d = self.dropout.backward(d)
        d = self.relu.backward(d)
        d = self.linear1.backward(d)
        return d

    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

    def set_training(self, flag):
        self.dropout.training = flag


class SentenceEmbedding:
    """Character-level tokenize -> embedding lookup -> + positional encoding -> dropout."""

    def __init__(self, max_sequence_length, d_model, char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob=0.1):
        self.max_sequence_length = max_sequence_length
        self.char_to_index = char_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.embedding = Embedding(len(char_to_index), d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = Dropout(drop_prob)

    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence):
            indices = [self.char_to_index[ch] for ch in list(sentence)]
            if start_token:
                indices.insert(0, self.char_to_index[self.START_TOKEN])
            if end_token:
                indices.append(self.char_to_index[self.END_TOKEN])
            indices = indices[: self.max_sequence_length]
            indices += [self.char_to_index[self.PADDING_TOKEN]] * (self.max_sequence_length - len(indices))
            return np.array(indices)

        return np.stack([tokenize(sentence) for sentence in batch])

    def forward(self, batch, start_token, end_token):
        tokens = self.batch_tokenize(batch, start_token, end_token)
        emb = self.embedding.forward(tokens)
        pos = self.position_encoder.forward()
        return self.dropout.forward(emb + pos)

    def backward(self, dout):
        d_emb_plus_pos = self.dropout.backward(dout)
        self.embedding.backward(d_emb_plus_pos)

    def parameters(self):
        return self.embedding.parameters()

    def set_training(self, flag):
        self.dropout.training = flag


class EncoderLayer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = Dropout(drop_prob)

    def forward(self, x, mask=None):
        residual = x
        a = self.dropout1.forward(self.attention.forward(x, mask))
        n1 = self.norm1.forward(a + residual)

        residual2 = n1
        f = self.dropout2.forward(self.ffn.forward(n1))
        n2 = self.norm2.forward(f + residual2)
        return n2

    def backward(self, dout):
        d_sum2 = self.norm2.backward(dout)          # gradient w.r.t. (f + residual2)
        df = self.dropout2.backward(d_sum2)
        dn1 = self.ffn.backward(df) + d_sum2          # residual branch gets a copy of d_sum2

        d_sum1 = self.norm1.backward(dn1)             # gradient w.r.t. (a + residual)
        da = self.dropout1.backward(d_sum1)
        dx = self.attention.backward(da) + d_sum1
        return dx

    def parameters(self):
        return (
            self.attention.parameters()
            + self.norm1.parameters()
            + self.ffn.parameters()
            + self.norm2.parameters()
        )

    def set_training(self, flag):
        self.dropout1.training = flag
        self.dropout2.training = flag
        self.ffn.set_training(flag)


class DecoderLayer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = Dropout(drop_prob)
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization(d_model)
        self.dropout3 = Dropout(drop_prob)

    def forward(self, x, y, self_mask, cross_mask):
        # x = encoder output (K/V source, fixed for this layer stack)
        # y = decoder state (Q source for cross-attention, updated each sub-layer)
        residual = y
        a = self.dropout1.forward(self.self_attention.forward(y, self_mask))
        n1 = self.norm1.forward(a + residual)

        residual2 = n1
        c = self.dropout2.forward(self.cross_attention.forward(x, n1, cross_mask))
        n2 = self.norm2.forward(c + residual2)

        residual3 = n2
        f = self.dropout3.forward(self.ffn.forward(n2))
        n3 = self.norm3.forward(f + residual3)
        return n3

    def backward(self, dout):
        d_sum3 = self.norm3.backward(dout)
        df = self.dropout3.backward(d_sum3)
        dn2 = self.ffn.backward(df) + d_sum3

        d_sum2 = self.norm2.backward(dn2)
        dc = self.dropout2.backward(d_sum2)
        dx_partial, dn1_from_cross = self.cross_attention.backward(dc)
        dn1 = dn1_from_cross + d_sum2

        d_sum1 = self.norm1.backward(dn1)
        da = self.dropout1.backward(d_sum1)
        dy = self.self_attention.backward(da) + d_sum1

        return dx_partial, dy

    def parameters(self):
        return (
            self.self_attention.parameters()
            + self.norm1.parameters()
            + self.cross_attention.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
            + self.norm3.parameters()
        )

    def set_training(self, flag):
        self.dropout1.training = flag
        self.dropout2.training = flag
        self.dropout3.training = flag
        self.ffn.set_training(flag)


class Encoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob
        )
        self.layers = [EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]

    def forward(self, x_batch, mask, start_token, end_token):
        x = self.sentence_embedding.forward(x_batch, start_token, end_token)
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x

    def backward(self, dout):
        dx = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        self.sentence_embedding.backward(dx)

    def parameters(self):
        params = list(self.sentence_embedding.parameters())
        for layer in self.layers:
            params += layer.parameters()
        return params

    def set_training(self, flag):
        self.sentence_embedding.set_training(flag)
        for layer in self.layers:
            layer.set_training(flag)


class Decoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob
        )
        self.layers = [DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]

    def forward(self, x, y_batch, self_mask, cross_mask, start_token, end_token):
        y = self.sentence_embedding.forward(y_batch, start_token, end_token)
        for layer in self.layers:
            y = layer.forward(x, y, self_mask, cross_mask)
        return y

    def backward(self, dout):
        """Returns the accumulated gradient w.r.t. the encoder output `x`."""
        dx_total = 0.0
        dy = dout
        for layer in reversed(self.layers):
            dx_partial, dy = layer.backward(dy)
            dx_total = dx_total + dx_partial
        self.sentence_embedding.backward(dy)
        return dx_total

    def parameters(self):
        params = list(self.sentence_embedding.parameters())
        for layer in self.layers:
            params += layer.parameters()
        return params

    def set_training(self, flag):
        self.sentence_embedding.set_training(flag)
        for layer in self.layers:
            layer.set_training(flag)


class Transformer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, target_vocab_size,
                 source_char_to_index, target_char_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.encoder = Encoder(
            d_model, ffn_hidden, num_heads, drop_prob, num_layers,
            max_sequence_length, source_char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN,
        )
        self.decoder = Decoder(
            d_model, ffn_hidden, num_heads, drop_prob, num_layers,
            max_sequence_length, target_char_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN,
        )
        self.linear = Linear(d_model, target_vocab_size)

    def forward(self, x_batch, y_batch, encoder_mask, decoder_self_mask, decoder_cross_mask,
                enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True):
        enc_out = self.encoder.forward(x_batch, encoder_mask, enc_start_token, enc_end_token)
        dec_out = self.decoder.forward(enc_out, y_batch, decoder_self_mask, decoder_cross_mask, dec_start_token, dec_end_token)
        return self.linear.forward(dec_out)

    def backward(self, dlogits):
        d_dec_out = self.linear.backward(dlogits)
        d_enc_out = self.decoder.backward(d_dec_out)
        self.encoder.backward(d_enc_out)

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters() + self.linear.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.encoder.set_training(True)
        self.decoder.set_training(True)

    def eval(self):
        self.encoder.set_training(False)
        self.decoder.set_training(False)


def cross_entropy_loss(logits, labels, pad_index):
    """logits: (B, S, V), labels: (B, S) int token ids. Padding positions are ignored."""
    B, S, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)
    probs = softmax_forward(logits_flat, axis=-1)

    valid = labels_flat != pad_index
    n_valid = max(int(valid.sum()), 1)
    row_idx = np.arange(labels_flat.shape[0])
    correct_probs = probs[row_idx, labels_flat]
    loss = -np.sum(np.log(np.clip(correct_probs, 1e-12, None)) * valid) / n_valid

    dlogits_flat = probs.copy()
    dlogits_flat[row_idx, labels_flat] -= 1
    dlogits_flat *= valid[:, None]
    dlogits_flat /= n_valid
    return loss, dlogits_flat.reshape(B, S, V)
