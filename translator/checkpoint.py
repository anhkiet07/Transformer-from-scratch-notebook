"""Save/load a trained Transformer's weights + vocab + hyperparams.

Uses pickle for simplicity (this is a local, trusted training artifact in
an educational project -- not something meant to load checkpoints from
untrusted sources).
"""

import pickle

from translator.data import END_TOKEN, PADDING_TOKEN, START_TOKEN
from translator.model import Transformer


def save_checkpoint(path, model, hyperparams, english_vocab, vietnamese_vocab):
    state = {
        "hyperparams": hyperparams,
        "english_vocab": english_vocab,
        "vietnamese_vocab": vietnamese_vocab,
        "params": [p.data.copy() for p in model.parameters()],
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        state = pickle.load(f)

    hp = state["hyperparams"]
    english_vocab = state["english_vocab"]
    vietnamese_vocab = state["vietnamese_vocab"]
    _, _, english_char_to_index = english_vocab
    vietnamese_vocabulary, _, vietnamese_char_to_index = vietnamese_vocab

    model = Transformer(
        hp["d_model"], hp["ffn_hidden"], hp["num_heads"], hp["drop_prob"], hp["num_layers"],
        hp["max_sequence_length"], len(vietnamese_vocabulary),
        english_char_to_index, vietnamese_char_to_index,
        START_TOKEN, END_TOKEN, PADDING_TOKEN,
    )
    for p, arr in zip(model.parameters(), state["params"]):
        p.data = arr.copy()

    model.eval()
    return model, hp, english_vocab, vietnamese_vocab
