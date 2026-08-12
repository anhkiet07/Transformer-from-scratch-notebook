# Known issues

Discrepancies and bugs found while reading `src/Encoder.py`, `src/Decoder.py`, `src/Transformer.py` closely. The goal here is to help anyone reading or reusing this code avoid surprises — not to criticize it; it's written for learning purposes, not production.

## 1. `MultiHeadAttention` is missing a transpose before reshape

**Location:** `Encoder.py:58`, `Decoder.py:96`, `Transformer.py:125` (`MultiHeadAttention.forward`, right before `out_linear`):

```python
values = values.reshape(batch_size, sequence_length, self.d_model)
```

Right before this line, `values` (the output of `scaled_dot_product_attention`) has shape `(B, num_heads, S, head_dim)` — the head axis still comes before the sequence axis (from the `transpose(0, 2, 1, 3)` earlier in the function). To correctly merge the heads back into `d_model`, you need to **transpose back** with `(0, 2, 1, 3)` → `(B, S, num_heads, head_dim)` **before** reshaping into `(B, S, d_model)`.

The current code reshapes directly from `(B, h, S, d_k)` to `(B, S, d_model)` without transposing first. Since the total element count matches (`h*S*d_k == S*d_model`), this line **does not raise an error**, but the data is read in the wrong order (NumPy reshape follows C-order) — head and position values get scrambled incorrectly. The result: self-attention's output is **not wrong in shape, but wrong in content** — the token at position `i` ends up receiving a partly incorrect mix of data belonging to other positions.

**Contrasting evidence:** `MultiHeadCrossAttention` in `Transformer.py:234-238` does it correctly:

```python
q = np.transpose(q, (0, 2, 1, 3))
kv = np.transpose(kv, (0, 2, 1, 3))
k, v = np.split(kv, 2, axis=-1)
values, attention = scaled_dot_product_attention(q, k, v, mask)
values = np.transpose(values, (0, 2, 1, 3)).reshape(batch_size, sequence_length, d_model)  # ← transposed back
```

**Impact:** this bug exists in all 3 files, everywhere `MultiHeadAttention` is used (self-attention in both the encoder and the decoder). Since there's no real backprop (see #4), this bug doesn't "break" any training run — but if someone adds an optimizer to actually train this model, it needs to be fixed first.

**Fix:** add `np.transpose(values, (0, 2, 1, 3))` before the `reshape` line, matching what `MultiHeadCrossAttention` (in `Transformer.py`) already does.

## 2. `Decoder.py`'s `MultiHeadCrossAttention` is missing a transpose for `kv`

**Location:** `Decoder.py:130-136`:

```python
kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # (B, S, h, 2*d_k)
q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)        # (B, S, h, d_k)
q = np.transpose(q, (0, 2, 1, 3))                                                 # (B, h, S, d_k)
k, v = np.split(kv, 2, axis=-1)   # ← kv is NOT transposed, still (B, S, h, d_k) each
values, attention = scaled_dot_product_attention(q, k, v, mask)
```

Here `q` has shape `(B, h, S, d_k)` but `k`/`v` are still shaped `(B, S, h, d_k)` — the two middle axes (`h` and `S`) are misaligned between `q` and `k`. Inside `scaled_dot_product_attention`, `np.matmul(q, swapaxes(k, -2, -1))` tries to broadcast the leading axes `(B, h)` against `(B, S)` — if `num_heads != sequence_length`, NumPy raises a broadcasting error (`ValueError: operands could not be broadcast together`) at runtime. If `num_heads` happens to equal `sequence_length`, the code runs but produces **wrong results** (same kind of axis-scrambling as issue #1) without raising any error.

**Comparison:** the `MultiHeadCrossAttention` version in `Transformer.py:235` transposes `kv` before splitting — correctly.

**Practical impact:** the project README states that the notebooks under `test/` import from `src/Transformer.py`, not `src/Decoder.py` directly, so this bug may never have been exercised end-to-end. If `src/Decoder.py` is used standalone (as its filename suggests — "standalone build to validate core mechanics"), calling `DecoderLayer.forward` will most likely raise an error at the cross-attention step.

**Fix:** add `kv = np.transpose(kv, (0, 2, 1, 3))` right after reshaping `kv`, before `np.split`, and transpose `values` back before the final `reshape` (same fix as issue #1).

## 3. `Encoder.py` (standalone) doesn't support masking

**Location:** `Encoder.py:123-132`, `EncoderLayer.forward(self, x)`:

```python
def forward(self, x):
    residual_x = x
    x = self.attention.forward(x, mask = None)   # always None — no way to pass one in
    ...
```

Unlike the version in `Transformer.py` (`forward(self, x, self_attention_mask)`), the standalone version has no way to pass in a padding mask — if `src/Encoder.py` is used on its own with a batch of differently-sized (padded) sentences, PAD tokens will participate in attention like real tokens, adding noise to the result. Not a bug (it's an intentional limitation of the "test attention/FFN/norm mechanics only" version), but worth knowing before using `Encoder.py` for anything beyond learning purposes.

## 4. No backward pass / optimizer

`Parameter` has a `.grad` field (`Encoder.py:20-23`, and similarly in the other two files), but **no class ever computes a gradient or assigns `.grad`**, and there's no optimizer (`SGD`, `Adam`, ...) anywhere in the repo. All of `src/` only runs a forward pass with fixed, randomly initialized weights — the current code cannot train a model on the data in `data/`. See also [transformer.md](transformer.md#learnable-parameters-parameter).

## 5. `PositionalEncoding` computes a redundant variable

**Location:** `Transformer.py:24-30`:

```python
even_i = np.arange(0, self.d_model, 2).astype(np.float32)
even_dominator = np.power(10000, even_i / self.d_model)

odd_i = np.arange(1, self.d_model, 2).astype(np.float32)
odd_dominator = np.power(10000, (odd_i - 1) / self.d_model)   # ← computed but unused

denominator = even_dominator   # only even_dominator is actually used
```

`odd_dominator` is mathematically always equal to `even_dominator` (since `(odd_i - 1) == even_i`, given `odd_i = even_i + 1`), so it's computed and then discarded. Not incorrect, just unpolished code — the two lines computing `odd_dominator` could be removed without changing the result. See [embedding.md](embedding.md#positionalencoding-transformerpy18-40).

## 6. Leftover debug `print()` calls

`MultiHeadAttention.forward`, `MultiHeadCrossAttention.forward` (in `Decoder.py`), and `DecoderLayer.forward` (in `Decoder.py`) all contain multiple `print(f"...")` calls that print shapes/step names to the console (e.g. `Decoder.py:156-177`: `"MASKED SELF ATTENTION"`, `"DROP OUT 1"`, ...). These are leftover debug traces from development — every `forward` call will print dozens of lines. Safe to remove if reusing this code, but useful to keep while first learning/debugging the shape flow.

## 7. `dec_start_token` defaults to `False` despite the author's own note that it should be `True`

**Location:** `Transformer.py:321`:

```python
dec_start_token=False, # We should make this true
```

In standard autoregressive inference, the decoder needs to start with a `<START>` token to generate the first output token. The `False` default means `SentenceEmbedding.forward` will not prepend `START_TOKEN` to the target sentence unless the caller explicitly passes `dec_start_token=True`. This is a leftover author's note (TODO), not a hidden bug — but it's easy to overlook when calling `Transformer.forward()` without setting this flag explicitly.

## 8. Character-level tokenization

Not a bug, but a notable design limitation: `SentenceEmbedding.batch_tokenize` (`Transformer.py:64-72`) uses `list(sentence)` — splitting **per character**, not per word or subword. For long sentences, this produces far more tokens than a subword tokenizer (BPE/WordPiece) used by real-world models would, requiring a larger `max_sequence_length` and making it harder for the model to learn word-level relationships. Fine for illustrating the architecture, not suited for achieving good translation quality. See [embedding.md](embedding.md#batch_tokenize-transformerpy62-78).
