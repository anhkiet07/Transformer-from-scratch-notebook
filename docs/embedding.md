# Embedding & Positional Encoding

Only present in `src/Transformer.py` (the standalone `Encoder.py`/`Decoder.py` take raw numeric tensors directly, with no tokenize/embed step). Three related classes: `Embedding`, `PositionalEncoding`, `SentenceEmbedding`.

## `Embedding` (`Transformer.py:42-47`)

```python
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) * 0.01

    def __call__(self, x):
        return self.weight[x]
```

A simple lookup table: a `(vocab_size, d_model)` matrix, randomly initialized to small values (`* 0.01`). `self.weight[x]` uses NumPy **fancy indexing** — given `x` (token ids) of shape `(B, S)`, the result is `(B, S, d_model)`, each token id replaced by its corresponding row vector in `weight`.

## `PositionalEncoding` (`Transformer.py:18-40`)

Since attention has no built-in notion of order (self-attention processes all positions in parallel and is permutation-invariant), positional information must be added to the embedding. Original formula from the paper:

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Implementation:

```python
def forward(self):
    even_i = np.arange(0, self.d_model, 2).astype(np.float32)      # 0, 2, 4, ...
    even_dominator = np.power(10000, even_i / self.d_model)

    odd_i = np.arange(1, self.d_model, 2).astype(np.float32)       # 1, 3, 5, ...
    odd_dominator = np.power(10000, (odd_i - 1) / self.d_model)

    denominator = even_dominator   # even_dominator == odd_dominator mathematically, see note below

    position = np.arange(self.max_sequence_length).reshape(-1, 1)  # (max_seq_len, 1)

    even_PE = np.sin(position / denominator)
    odd_PE  = np.cos(position / denominator)

    stacked = np.stack((even_PE, odd_PE), axis=2)   # (max_seq_len, d_model/2, 2)
    PE = stacked.reshape(stacked.shape[0], -1)       # (max_seq_len, d_model) — interleaved sin, cos, sin, cos...
    return PE
```

**Note:** `even_dominator` and `odd_dominator` are computed as two separate variables but are mathematically always equal — `even_i / d_model = (2i)/d_model` and `(odd_i - 1)/d_model = (2i+1-1)/d_model = 2i/d_model` are the same expression. The code sets `denominator = even_dominator`, so `odd_dominator` is computed but never used — redundant, but not incorrect, just not simplified.

`np.stack(..., axis=2)` followed by `reshape` is a trick to interleave the two arrays `sin` and `cos` into the correct order `[sin₀, cos₀, sin₁, cos₁, ...]` along the last axis — equivalent to directly assigning `PE[:, 0::2] = sin(...)`, `PE[:, 1::2] = cos(...)` but without pre-allocating the array.

Result: a `PE` matrix of shape `(max_sequence_length, d_model)`, added directly to the embedding (it has no learnable parameters — this is **fixed/sinusoidal** positional encoding, not a learned positional embedding).

## `SentenceEmbedding` (`Transformer.py:49-85`)

The assembly class: text sentence (string) → token id → embedding vector → add positional encoding → dropout.

```python
class SentenceEmbedding:
    def __init__(self, max_sequence_length, d_model, language_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.vocab_size = len(language_to_index)
        self.embedding = Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = Dropout(p=0.1)
```

### `batch_tokenize` (`Transformer.py:62-78`)

```python
def tokenize(sentence, start_token, end_token):
    sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
    if start_token: sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:   sentence_word_indicies.append(language_to_index[END_TOKEN])
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return np.array(sentence_word_indicies)
```

Notable: `list(sentence)` splits a string into **individual characters** (character-level tokenization), not words — every element of `list("hello")` is `'h', 'e', 'l', 'l', 'o'`. This is the simplest tokenizer possible (no word vocabulary, no BPE/subword needed), fitting for an academic "from scratch" implementation, but much less efficient than the subword tokenization used by real-world models.

After tokenizing, the sentence is padded to exactly `max_sequence_length` with `PADDING_TOKEN` — necessary to stack sentences of different lengths into a single rectangular batch tensor.

### `forward` (`Transformer.py:80-85`)

```python
def forward(self, x, start_token, end_token):
    x = self.batch_tokenize(x, start_token, end_token)   # (B, S) — token ids
    x = self.embedding(x)                                  # (B, S, d_model)
    pos = self.position_encoder.forward()                  # (max_seq_len, d_model)
    x = self.dropout(x + pos)                               # broadcasts over batch → (B, S, d_model)
    return x
```

`x + pos` broadcasts `pos` (no batch axis) over every sentence in the batch — every sentence shares the same positional encoding table, since position within a sentence is an absolute concept, independent of sentence content.

**Note:** `pos` has shape `(max_sequence_length, d_model)` while `x` has shape `(B, max_sequence_length, d_model)` (every sentence is padded to `max_sequence_length`), so the broadcasted addition is valid. If this were later changed to skip padding to a fixed `max_sequence_length`, `pos[:S]` would need to be sliced before adding.
