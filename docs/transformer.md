# Transformer — assembling the full model

Defined in `src/Transformer.py:294-326`. This is the top-level class, combining `Encoder` + `Decoder` + a final `Linear` layer projecting onto the target vocabulary, forming an English → Kannada machine-translation pipeline (per the variable names `english_to_index`, `kannada_to_index`, `kn_vocab_size` — though the architecture is general enough for any language pair).

## Initialization

```python
class Transformer:
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, kn_vocab_size,
                 english_to_index, kannada_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN):
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                                max_sequence_length, english_to_index,
                                START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers,
                                max_sequence_length, kannada_to_index,
                                START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = Linear(d_model, kn_vocab_size)
```

Worth noting: the encoder and decoder use **two different vocabularies** (`english_to_index` vs `kannada_to_index`) — each language gets its own `SentenceEmbedding`/`Embedding` (no shared weights), while sharing the same architectural hyperparameters (`d_model`, `num_heads`, `num_layers`, ...).

## `forward`

```python
def forward(self, x, y,
            encoder_self_attention_mask=None,
            decoder_self_attention_mask=None,
            decoder_cross_attention_mask=None,
            enc_start_token=False, enc_end_token=False,
            dec_start_token=False, dec_end_token=False):
    x = self.encoder.forward(x, encoder_self_attention_mask,
                              start_token=enc_start_token, end_token=enc_end_token)
    out = self.decoder.forward(x, y, decoder_self_attention_mask, decoder_cross_attention_mask,
                                start_token=dec_start_token, end_token=dec_end_token)
    out = self.linear(out)
    return out
```

- `x`: a batch of English sentences (list[str]).
- `y`: a batch of Kannada sentences (list[str]) — in real training (teacher forcing) this is the target sentence (ground truth); the decoder learns to predict the next token from the previous tokens of this same sentence, combined with `decoder_self_attention_mask` so it can't see future tokens.
- The final output `out` has shape `(B, S, kn_vocab_size)` — **pre-softmax logits** for every position; to get a probability distribution over the vocabulary you must apply `softmax(out, axis=-1)` yourself (not done inside `forward`), and for training, `cross_entropy` is typically applied directly on the logits (softmax is folded into the loss function in most frameworks).
- All 3 mask parameters default to `None`: if not passed, nothing is masked — only sensible if every sentence in the batch has the same true length (no padding mask needed) and no causal mask is applied (not accurate for real decoder training — see the note below).
- The `dec_start_token=False` flag has an inline comment `# We should make this true` right in the code (`Transformer.py:321`) — the author's own note that the current default is not the intended behavior; in real usage `dec_start_token=True` should be passed so the decoder input gets a `START` token prepended (standard autoregressive decoding: the decoder must begin from `<START>`, not the first token of the target sentence).

## Building masks (reference only — not present in `src/`)

`src/Transformer.py` accepts masks as parameters but **includes no function to build them**. In real usage you'd need to construct them yourself, e.g.:

```python
NEG_INFTY = -1e9

def create_masks(eng_batch, kn_batch, max_sequence_length, PADDING_TOKEN):
    look_ahead_mask = np.triu(np.full((max_sequence_length, max_sequence_length), NEG_INFTY), k=1)
    # ... combine with a padding mask based on each sentence's true length ...
```

This is the "left as an exercise" part of the project — the notebooks under `test/` may demonstrate calling `Transformer.forward` with `mask=None` (skipping masking entirely) just to check output shapes, which doesn't necessarily represent how the model should actually be trained.

## Learnable parameters (`Parameter`)

Every `Linear` and `LayerNormalization` creates `Parameter` objects (with `.data` and `.grad`), but there is **no optimizer or `backward()` function** anywhere in `src/` — `.grad` is always `None`, never computed or used to update `.data`. In other words, the model only runs a forward pass with fixed random initial weights; actual training would require implementing backprop yourself (or porting this to a framework with autograd, like PyTorch). Important to understand before expecting the model to "learn" anything from the data in `data/`.

## Trying it out

```python
from src.Transformer import Transformer

transformer = Transformer(
    d_model=512, ffn_hidden=2048, num_heads=8, drop_prob=0.1, num_layers=1,
    max_sequence_length=200, kn_vocab_size=len(kannada_vocab),
    english_to_index=english_to_index, kannada_to_index=kannada_to_index,
    START_TOKEN="<START>", END_TOKEN="<END>", PADDING_TOKEN="<PAD>",
)

out = transformer.forward(
    ["hello world"], ["ನಮಸ್ಕಾರ"],
    enc_start_token=True, enc_end_token=True,
    dec_start_token=True, dec_end_token=True,
)
print(out.shape)  # (1, max_sequence_length, kn_vocab_size)
```

See `test/test transformer.ipynb` for a working example against real data from `data/`.
