# Decoder

Related files: `src/Decoder.py` (standalone build) and the `Decoder`/`DecoderLayer`/`SequentialDecoder` section in `src/Transformer.py:242-292` (the full build). The decoder has **3 sub-layers** per layer (vs. 2 for the encoder): masked self-attention, cross-attention, feed-forward.

## `DecoderLayer` — one decoder layer

### Standalone version (`Decoder.py:141-178`)

```python
class DecoderLayer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        self.self_attention = MultiHeadAttention(d_model, d_model, num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = Dropout(drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization([d_model])
        self.dropout3 = Dropout(drop_prob)

    def forward(self, x, y, decoder_mask):
        # x = encoder output, y = decoder input (target sequence)
        _y = y
        y = self.self_attention.forward(y, mask=decoder_mask)   # (1) masked self-attention
        y = self.dropout1(y)
        y = self.norm1.forward(y + _y)                            # Add & Norm #1

        _y = y
        y = self.encoder_decoder_attention.forward(x, y, mask=None)  # (2) cross-attention: Q=y, K/V=x
        y = self.dropout2(y)
        y = self.norm2.forward(y + _y)                             # Add & Norm #2

        _y = y
        y = self.ffn.forward(y)                                    # (3) feed-forward
        y = self.dropout3(y)
        y = self.norm3.forward(y + _y)                              # Add & Norm #3
        return y
```

The three blocks are clear from the variable names: `y` is always the decoder state (updated at each step), and `x` is the encoder output (fixed, read-only — used as K/V for cross-attention).

### Version in `Transformer.py` (`Transformer.py:242-270`)

Structurally identical, differing in 2 ways:

```python
def forward(self, x, y, self_attention_mask, cross_attention_mask):
    ...
    y = self.self_attention.forward(y, mask=self_attention_mask)
    ...
    y = self.encoder_decoder_attention.forward(x, y, mask=cross_attention_mask)
    ...
```

1. It takes **two separate masks**: `self_attention_mask` (causal/look-ahead mask for self-attention) and `cross_attention_mask` (padding mask for cross-attention against the encoder output) — instead of the standalone version's single mask, with cross-attention always using `mask=None`.
2. No debug `print()` calls.

## Why does decoder self-attention need masking ("masked self-attention")?

During training, the decoder is shown the **entire** target sentence at once (teacher forcing) so computation can be parallelized, but at position `t` the model is only allowed to use information from positions `≤ t` — otherwise it would "see" the correct token it's supposed to predict and learn a meaningless shortcut that doesn't hold at real inference time (when future tokens don't exist yet). The `decoder_mask`/`self_attention_mask` is an upper-triangular matrix filled with `-∞` (or a very negative number) at positions `j > i`, added to the attention scores before softmax to zero out the weight of future positions — see the formula in [attention.md](attention.md#1-scaled_dot_product_attentionq-k-v-maskNone).

## Why doesn't cross-attention need a causal mask (only a padding mask)?

Cross-attention reads from the **encoder output**, which is already fixed and fully known ahead of time (there's no "peeking at the future" problem, since the source sentence isn't being generated sequentially). The `cross_attention_mask` in the `Transformer.py` version only needs to hide PAD positions in the source sentence.

## `SequentialDecoder`

### Standalone version (`Decoder.py:180-188`)

```python
class SequentialDecoder:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, *inputs):
        x, y, mask = inputs
        for layer in self.layers:
            y = layer.forward(x, y, mask)
        return y
```

### Version in `Transformer.py` (`Transformer.py:272-280`)

```python
def forward(self, *inputs):
    x, y, self_attention_mask, cross_attention_mask = inputs
    for layer in self.layers:
        y = layer.forward(x, y, self_attention_mask, cross_attention_mask)
    return y
```

Like `SequentialEncoder`: a dedicated wrapper is needed because each layer's `forward` takes more than one dynamic argument (`x` stays fixed across layers, `y` is updated each layer, plus one or two masks).

## `Decoder` — the full stack

### Standalone version (`Decoder.py:190-197`)

```python
class Decoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        self.layers = SequentialDecoder(*[DecoderLayer(...) for _ in range(num_layers)])

    def forward(self, x, y, mask):
        return self.layers.forward(x, y, mask)
```

Takes `x` (encoder output) and `y` (decoder input) directly as tensors — no separate embedding step.

### Version in `Transformer.py` (`Transformer.py:282-292`)

```python
class Decoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(...)
        self.layers = SequentialDecoder(*[DecoderLayer(...) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding.forward(y, start_token, end_token)  # target sentence (text) → tensor
        y = self.layers.forward(x, y, self_attention_mask, cross_attention_mask)
        return y
```

`x` (encoder output) is already a numeric tensor (it comes from `Encoder.forward`), only `y` (the target sentence) needs tokenizing/embedding, since it's still raw text when it enters `Transformer.forward`.

## Quick comparison of the two Decoder versions

| | `src/Decoder.py` | `Decoder` in `src/Transformer.py` |
|---|---|---|
| Input `y` | numeric tensor | list of text sentences (str) |
| Embedding for `y` | none | yes (`SentenceEmbedding`) |
| Number of masks | 1 (used for self-attention; cross-attention always `None`) | 2 (`self_attention_mask` + `cross_attention_mask`) |
| Default `num_layers` | `1` | required (no default) |

See also the shape bug in `Decoder.py`'s `MultiHeadCrossAttention`, described in [known-issues.md #2](known-issues.md#2-decoderpys-multiheadcrossattention-is-missing-a-transpose-for-kv).
