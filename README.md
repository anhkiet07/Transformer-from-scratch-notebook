# Transformer from Scratch

<p align="left">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/dependencies-NumPy%20only-orange" alt="NumPy only">
  <img src="https://img.shields.io/badge/autograd-none%20(forward%20pass%20only)-lightgrey" alt="Forward pass only">
</p>

A from-scratch implementation of the Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017) — no PyTorch, no TensorFlow, just NumPy. Every building block is first derived step by step in a notebook, then assembled into a working, testable implementation under `src/`.

## Table of contents

- [Project structure](#project-structure)
- [Learning path](#learning-path)
- [Source code](#source-code)
- [Architecture](#architecture)
- [Tests](#tests)
- [Getting started](#getting-started)
- [License](#license)

## Project structure

```text
Transformer/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── Self Attention.ipynb
│   ├── Multi-head Attention.ipynb
│   ├── Multi-head Cross Attention.ipynb
│   ├── Layer Norm.ipynb
│   ├── Position-wise Feed Forward.ipynb
│   ├── Positional Encoding .ipynb
│   └── Sentence Embedding.ipynb
├── src/
│   ├── __init__.py
│   ├── Encoder.py
│   ├── Decoder.py
│   └── Transformer.py
└── test/
    ├── test encoder.ipynb
    ├── test decoder.ipynb
    └── test transformer.ipynb
```

## Learning path

The notebooks build up the architecture one concept at a time. Suggested reading order:

| # | Notebook | Concept |
|---|----------|---------|
| 1 | `Self Attention.ipynb` | Scaled dot-product attention, from Q/K/V to softmax weights |
| 2 | `Multi-head Attention.ipynb` | Splitting attention into multiple heads and reshaping tensors |
| 3 | `Layer Norm.ipynb` | Layer normalization over the feature axis |
| 4 | `Position-wise Feed Forward.ipynb` | The two-layer FFN applied to every position |
| 5 | `Positional Encoding .ipynb` | Injecting sequence order via sinusoidal encodings |
| 6 | `Sentence Embedding.ipynb` | Tokenizing sentences and turning them into embeddings |
| 7 | `Multi-head Cross Attention.ipynb` | Attention between encoder output and decoder input |

## Source code

Everything in `src/` is pure NumPy — no autograd, no GPU, forward pass only.

| File | Contents |
|------|----------|
| `Transformer.py` | Self-contained, fully wired implementation: `Embedding`, `PositionalEncoding`, `SentenceEmbedding`, `Linear`, `MultiHeadAttention`, `MultiHeadCrossAttention`, `LayerNormalization`, `PositionwiseFeedForward`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, and `Transformer`. |
| `Encoder.py` | Standalone build of the encoder stack (attention + FFN + norm), used to validate the core mechanics before embeddings were wired in. |
| `Decoder.py` | Standalone build of the decoder stack, including masked self-attention and encoder-decoder cross-attention. |

## Architecture

```text
source sentences                          target sentences
       │                                          │
       ▼                                          ▼
 SentenceEmbedding                         SentenceEmbedding
 (tokenize + embed + positional encoding)  (tokenize + embed + positional encoding)
       │                                          │
       ▼                                          │
 ┌───────────────┐  N ×                            │
 │  EncoderLayer  │                                 │
 │  self-attn     │                                 │
 │  add & norm    │                                 │
 │  feed forward  │                                 │
 │  add & norm    │                                 │
 └───────┬───────┘                                 │
         │ encoder output                          ▼
         │                                 ┌────────────────┐  N ×
         └───────────────────────────────► │  DecoderLayer   │
                                            │  masked self-attn│
                                            │  add & norm       │
                                            │  cross-attn (↑)   │
                                            │  add & norm        │
                                            │  feed forward       │
                                            │  add & norm           │
                                            └──────────┬───────────┘
                                                        ▼
                                                Linear (→ vocab size)
                                                        ▼
                                                     logits
```

## Tests

Each test notebook in `test/` imports directly from `src/Transformer.py`, runs a forward pass on random/toy data, and asserts the output shape:

- `test encoder.ipynb` — runs the encoder stack on a random input tensor.
- `test decoder.ipynb` — runs the decoder against a fake encoder output and a batch of sentences.
- `test transformer.ipynb` — runs the full encoder → decoder → linear pipeline end to end.

## Getting started

1. Install the only dependency:
   ```bash
   pip install numpy jupyter
   ```
2. Open the `Transformer` folder in VS Code (or run `jupyter notebook`).
3. To follow the derivation, work through `notebooks/` in the order above.
4. To see the assembled architecture run, open any notebook in `test/` and run all cells top to bottom.

## License

Released under the [MIT License](LICENSE) — Copyright (c) 2026 Nguyễn Anh Kiệt.
