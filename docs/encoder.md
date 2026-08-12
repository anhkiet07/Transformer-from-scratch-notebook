# Encoder

Related files: `src/Encoder.py` (standalone build) and the `Encoder`/`EncoderLayer`/`SequentialEncoder` section in `src/Transformer.py:179-216` (the full build, with embeddings + masks). Know which version you're looking at — the APIs differ; see the comparison table below.

## `LayerNormalization` (shared, `Encoder.py:75-89`)

```python
class LayerNormalization:
    def __init__(self, parameters_shape, eps=1e-5):
        self.gamma = np.ones(parameters_shape)   # scale — learnable (no actual update step in this repo)
        self.beta = np.zeros(parameters_shape)   # shift — learnable

    def forward(self, inputs):
        dims = tuple(-(i + 1) for i in range(len(self.parameters_shape)))  # the trailing feature axis
        mean = inputs.mean(axis=dims, keepdims=True)
        var  = ((inputs - mean) ** 2).mean(axis=dims, keepdims=True)
        std  = np.sqrt(var + self.eps)
        y = (inputs - mean) / std
        return self.gamma * y + self.beta
```

Normalizes over the **feature** axis (the `d_model` dimension), computed per token — unlike Batch Normalization (which normalizes over the batch axis). Formula: `y = γ · (x - μ) / √(σ² + ε) + β`, with `μ, σ²` computed over each token's `d_model`-dimensional vector. `eps` avoids division by zero when the variance is near zero.

## `PositionwiseFeedForward` (shared, `Encoder.py:91-112`)

```python
class PositionwiseFeedForward:
    def __init__(self, d_model, hidden_units, drop_prop):
        self.linear1 = Linear(d_model, hidden_units)
        self.linear2 = Linear(hidden_units, d_model)

    def forward(self, x):
        x = self.linear1(x)   # (B, S, d_model) → (B, S, hidden_units)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)   # (B, S, hidden_units) → (B, S, d_model)
        return x
```

A 2-layer feed-forward network (`Linear → ReLU → Dropout → Linear`) applied **independently to every position** (position-wise) — since `Linear` only multiplies along the last axis, every token in `(B, S, d_model)` goes through the exact same transformation without interacting with other tokens (unlike attention, where tokens exchange information). `hidden_units` is usually larger than `d_model` (e.g. 2048 vs. 512 in the original paper) to increase nonlinear representational capacity.

## `EncoderLayer` — one encoder layer

### Standalone version (`Encoder.py:114-132`)

```python
class EncoderLayer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prop):
        self.attention = MultiHeadAttention(input_dim=d_model, d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.dropout1 = Dropout(drop_prop)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prop)
        self.norm2 = LayerNormalization([d_model])
        self.dropout2 = Dropout(drop_prop)

    def forward(self, x):
        residual_x = x
        x = self.attention.forward(x, mask=None)     # mask always None — no mask parameter accepted
        x = self.dropout1(x)
        x = self.norm1.forward(x + residual_x)        # Add & Norm #1

        residual_x = x
        x = self.ffn.forward(x)
        x = self.dropout2(x)
        x = self.norm2.forward(x + residual_x)         # Add & Norm #2
        return x
```

### Version in `Transformer.py` (`Transformer.py:179-197`)

Structurally identical, with one difference: `forward(self, x, self_attention_mask)` — it **accepts and forwards a real mask**, into `self.attention.forward(x, mask=self_attention_mask)`, allowing a padding mask to be applied when encoding sentences of varying length within the same batch.

### Post-norm architecture

Both versions follow the **post-norm** pattern (normalize *after* adding the residual): `norm(sublayer(x) + x)`, matching the original 2017 paper (unlike the "pre-norm" variants more common in modern models, where normalization is applied *before* the sublayer).

## `SequentialEncoder` (only in `Transformer.py:198-205`)

```python
class SequentialEncoder:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, self_attention_mask):
        for layer in self.layers:
            x = layer.forward(x, self_attention_mask)
        return x
```

A thin wrapper to pass `x` **and** `self_attention_mask` through multiple `EncoderLayer`s in sequence — needed because `EncoderLayer.forward` takes 2 arguments, whereas a plain `for` loop (as in `Encoder.py`) can only forward `x`.

## `Encoder` — the full stack

### Standalone version (`Encoder.py:134-142`)

```python
class Encoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        self.layers = [EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                        for _ in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

Takes a raw numeric tensor `(B, S, d_model)` directly — the caller is expected to prepare the embedding externally. No mask, no embedding.

### Version in `Transformer.py` (`Transformer.py:206-216`)

```python
class Encoder:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.sentence_embedding = SentenceEmbedding(...)
        self.layers = SequentialEncoder(*[EncoderLayer(...) for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding.forward(x, start_token, end_token)   # text sentence → tensor
        x = self.layers.forward(x, self_attention_mask)                   # through N encoder layers
        return x
```

Takes **text sentences** directly (`x` is a list of strings), tokenizing and embedding internally before running the stack. This is the actual entry point used when running `Transformer.forward`.

## Quick comparison of the two Encoder versions

| | `src/Encoder.py` | `Encoder` in `src/Transformer.py` |
|---|---|---|
| Input | numeric tensor `(B, S, d_model)` | list of text sentences (str) |
| Embedding | none (assumes input is already a vector) | yes (`SentenceEmbedding`) |
| Positional encoding | none | yes |
| Mask | not supported (`mask=None` hardcoded) | supported (`self_attention_mask` forwarded through every layer) |
| When to use | testing attention/FFN/norm mechanics in isolation | running the full machine-translation pipeline |
