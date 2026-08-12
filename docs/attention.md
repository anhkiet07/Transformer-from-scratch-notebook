# Attention

Three attention mechanisms appear in `src/`:

1. `scaled_dot_product_attention` — the core math function, shared by every attention type.
2. `MultiHeadAttention` — multi-head self-attention (Q, K, V all come from the same input).
3. `MultiHeadCrossAttention` — multi-head cross-attention (Q from the decoder, K/V from the encoder).

## 1. `scaled_dot_product_attention(q, k, v, mask=None)`

Defined at the top of all 3 files (`Encoder.py:10-17`, `Decoder.py:10-17`, `Transformer.py:9-16`):

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scaled = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled, axis=-1)
    out = np.matmul(attention, v)
    return out, attention
```

Corresponding formula:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k + mask ) · V
```

- `d_k = q.shape[-1]`: the dimension of a single head (not `d_model`). Dividing by `√d_k` keeps the variance of the dot product stable as `d_k` grows, preventing softmax from saturating (gradients collapsing to ~0).
- `mask`: added **before** softmax, typically a matrix of all `0` (keep) or `-∞`/a very negative number (drop that position from attention). Used for:
  - **Padding mask**: hides PAD positions in a sentence.
  - **Look-ahead mask** (causal mask): hides future positions in the decoder's self-attention so the model can't "peek" at tokens it hasn't generated yet.
- Returns both `out` (the values blended by attention weight) and `attention` (the weight matrix itself, useful for visualization/debugging).

### `softmax(x, axis=-1)`

```python
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

Subtracting `max` before `exp` is the standard **numerically stable softmax** trick — it avoids overflow when inputs are large, without changing the result, since `softmax(x) = softmax(x - c)` for any constant `c`.

## 2. `MultiHeadAttention` (self-attention)

Defined in `Encoder.py:35-61`, `Decoder.py:73-99`, `Transformer.py:102-128` (3 near-identical copies).

```python
class MultiHeadAttention:
    def __init__(self, input_dim, d_model, num_heads):
        self.head_dim = d_model // num_heads
        self.qkv_linear = Linear(input_dim, d_model * 3)   # one projection for Q, K and V
        self.out_linear = Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_linear(x)                                            # (B, S, 3*d_model)
        qkv = qkv.reshape(B, S, num_heads, 3*head_dim)                      # (B, S, h, 3*d_k)
        qkv = np.transpose(qkv, (0, 2, 1, 3))                               # (B, h, S, 3*d_k)
        q, k, v = np.split(qkv, 3, axis=-1)                                 # each (B, h, S, d_k)
        values, attention = scaled_dot_product_attention(q, k, v, mask)     # values: (B, h, S, d_k)
        values = values.reshape(batch_size, sequence_length, self.d_model)  # (B, S, d_model)
        out = self.out_linear(values)
        return out
```

**Why a single `Linear` for Q, K and V?** Instead of 3 separate projections (`W_Q`, `W_K`, `W_V`), this implementation packs them into one `qkv_linear` that projects `input_dim → 3 * d_model` and then splits the result in three. Mathematically equivalent to 3 separate projections, but more efficient (one large matmul instead of three small ones).

**Why `transpose` before `split`?** After `reshape` the tensor is `(B, S, h, 3*d_k)` — the head axis sits between the batch and sequence axes. `transpose(0, 2, 1, 3)` moves it to `(B, h, S, 3*d_k)` so each head becomes an independent "batch"; this lets `scaled_dot_product_attention` run over all heads in parallel via `np.matmul` (matmul broadcasts over the leading axes and only multiplies the last two).

**⚠️ Note on the `.reshape` right after attention:** `values = values.reshape(batch_size, sequence_length, self.d_model)` is called **immediately after** `scaled_dot_product_attention`, while `values` is still laid out as `(B, h, S, d_k)` — it has not been transposed back to `(B, S, h, d_k)`. See [known-issues.md #1](known-issues.md#1-multiheadattention-is-missing-a-transpose-before-reshape) for why this is a bug and what it does.

**The `print()` calls in `forward`:** every step of `MultiHeadAttention.forward` and `DecoderLayer.forward` has a `print(f"...shape...")` call. These are leftover debug traces from development (useful for watching shapes evolve step by step while first writing the code) — not intentional logging, and they will print dozens of lines on every `forward` call. Safe to remove if reusing this code elsewhere.

## 3. `MultiHeadCrossAttention`

Defined in `Decoder.py:113-139`, `Transformer.py:218-240`. **Does not exist in `Encoder.py`** (the encoder never needs cross-attention).

The core difference from self-attention: Q comes from one source (`y`, the decoder), while K/V come from a different source (`x`, the encoder output).

```python
class MultiHeadCrossAttention:
    def __init__(self, d_model, num_heads):
        self.kv_layer = Linear(d_model, 2 * d_model)  # projects K, V from the encoder output
        self.q_layer  = Linear(d_model, d_model)       # projects Q from the decoder state

    def forward(self, x, y, mask=None):
        # x = encoder output  (B, S, d_model)  → source of K, V
        # y = decoder state   (B, S, d_model)  → source of Q
        kv = self.kv_layer(x)
        q  = self.q_layer(y)
        ...
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        ...
```

Intuition: at every position being decoded, the decoder "queries" information from the entire encoder output sequence (key/value) to decide which part of the source sentence to attend to when generating the next token — this is the classic encoder-decoder attention mechanism that predates the Transformer (Bahdanau/Luong attention), generalized here into a multi-head form.

The version in `Decoder.py` and the version in `Transformer.py` are **not identical** — `Decoder.py`'s version is missing a `transpose` step for `kv` and a transpose-back step for `values`, producing wrong shapes. Details in [known-issues.md #2](known-issues.md#2-decoderpys-multiheadcrossattention-is-missing-a-transpose-for-kv).

## Shape summary through Multi-Head (Cross-)Attention

| Step | Shape |
|---|---|
| Input `x` (or `x`, `y` for cross-attn) | `(B, S, d_model)` |
| After `qkv_linear` / `q_layer`+`kv_layer` | `(B, S, 3*d_model)` or split into `q:(B,S,d_model)`, `kv:(B,S,2*d_model)` |
| After reshaping into heads | `(B, S, h, d_k)` (×3 or ×1/×2) |
| After transpose | `(B, h, S, d_k)` |
| After `scaled_dot_product_attention` | `values: (B, h, S, d_k)`, `attention: (B, h, S, S)` |
| After transpose-back + reshape | `(B, S, d_model)` |
| After `out_linear`/`linear_layer` | `(B, S, d_model)` |
