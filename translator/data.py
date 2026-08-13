"""
Loads data/train.en + data/train.vi, cleans them, and builds small
character-level vocabularies -- sized for a CPU-only NumPy training run
(a few hundred short sentence pairs), not the full 133k-sentence corpus.
"""

import os
import re
import unicodedata

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
ENGLISH_FILE = os.path.join(DATA_DIR, "train.en")
VIETNAMESE_FILE = os.path.join(DATA_DIR, "train.vi")

START_TOKEN = "<START>"
PADDING_TOKEN = "<PAD>"
END_TOKEN = "<END>"

MOSES_ESCAPES = re.compile(r"&(apos|quot|amp|lt|gt);")


def clean_sentence(sentence):
    """Strip everything except letters, digits and whitespace.

    NFC-normalizes first so Vietnamese diacritics are single letter
    codepoints rather than separate combining marks, and replaces
    Moses-style escapes ("&apos;") with a space before filtering --
    otherwise "i &apos;d" would collapse into "iapos d".
    """
    sentence = MOSES_ESCAPES.sub(" ", sentence)
    sentence = unicodedata.normalize("NFC", sentence)
    kept = [ch for ch in sentence if ch.isalpha() or ch.isdigit() or ch.isspace()]
    cleaned = "".join(kept)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_vocabulary(sentences):
    charset = set()
    for sentence in sentences:
        charset.update(sentence)
    vocabulary = [START_TOKEN, PADDING_TOKEN, END_TOKEN] + sorted(charset)
    index_to_char = dict(enumerate(vocabulary))
    char_to_index = {ch: i for i, ch in enumerate(vocabulary)}
    return vocabulary, index_to_char, char_to_index


def load_dataset(num_sentences=300, max_sentence_length=30, scan_limit=50000, seed=0):
    """Returns (english_sentences, vietnamese_sentences, max_sequence_length,
    english_vocab, vietnamese_vocab) where each *_vocab is
    (vocabulary, index_to_char, char_to_index).

    Scans the first `scan_limit` lines of train.en/train.vi, keeps pairs
    where both sentences (after cleaning) are non-empty and no longer than
    `max_sentence_length` characters, and takes the first `num_sentences`
    of those -- small and short by design, for a training run that
    actually finishes on CPU-only NumPy.
    """
    with open(ENGLISH_FILE, "r", encoding="utf-8") as f:
        english_raw = f.readlines()[:scan_limit]
    with open(VIETNAMESE_FILE, "r", encoding="utf-8") as f:
        vietnamese_raw = f.readlines()[:scan_limit]

    english_clean = [clean_sentence(s.rstrip("\n").lower()) for s in english_raw]
    vietnamese_clean = [clean_sentence(s.rstrip("\n").lower()) for s in vietnamese_raw]

    english_sentences, vietnamese_sentences = [], []
    for en, vi in zip(english_clean, vietnamese_clean):
        if not en or not vi:
            continue
        if len(en) > max_sentence_length or len(vi) > max_sentence_length:
            continue
        english_sentences.append(en)
        vietnamese_sentences.append(vi)
        if len(english_sentences) >= num_sentences:
            break

    max_sequence_length = max_sentence_length + 2  # room for START and END tokens

    english_vocab = build_vocabulary(english_sentences)
    vietnamese_vocab = build_vocabulary(vietnamese_sentences)

    return english_sentences, vietnamese_sentences, max_sequence_length, english_vocab, vietnamese_vocab


def iterate_batches(english_sentences, vietnamese_sentences, batch_size, shuffle=True, seed=0):
    n = len(english_sentences)
    order = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        batch_idx = order[start : start + batch_size]
        eng_batch = tuple(english_sentences[i] for i in batch_idx)
        vi_batch = tuple(vietnamese_sentences[i] for i in batch_idx)
        yield eng_batch, vi_batch


NEG_INFTY = -1e9


def create_masks(eng_batch, vi_batch, max_sequence_length):
    num_sentences = len(eng_batch)
    look_ahead_mask = np.triu(np.full((max_sequence_length, max_sequence_length), True), k=1)

    encoder_padding_mask = np.full((num_sentences, max_sequence_length, max_sequence_length), False)
    decoder_padding_mask_self = np.full((num_sentences, max_sequence_length, max_sequence_length), False)
    decoder_padding_mask_cross = np.full((num_sentences, max_sequence_length, max_sequence_length), False)

    for idx in range(num_sentences):
        eng_len, vi_len = len(eng_batch[idx]), len(vi_batch[idx])
        # +1 accounts for the END token appended to each sequence.
        eng_pad = np.arange(eng_len + 1, max_sequence_length)
        vi_pad = np.arange(vi_len + 1, max_sequence_length)

        encoder_padding_mask[idx, :, eng_pad] = True
        encoder_padding_mask[idx, eng_pad, :] = True
        decoder_padding_mask_self[idx, :, vi_pad] = True
        decoder_padding_mask_self[idx, vi_pad, :] = True
        decoder_padding_mask_cross[idx, :, eng_pad] = True
        decoder_padding_mask_cross[idx, vi_pad, :] = True

    encoder_self_attention_mask = np.where(encoder_padding_mask, NEG_INFTY, 0.0)
    decoder_self_attention_mask = np.where(look_ahead_mask | decoder_padding_mask_self, NEG_INFTY, 0.0)
    decoder_cross_attention_mask = np.where(decoder_padding_mask_cross, NEG_INFTY, 0.0)

    # Attention scores are shaped (B, num_heads, S, S); insert a size-1 head
    # axis so these (B, S, S) masks broadcast correctly against them
    # regardless of num_heads (broadcasting B against num_heads directly,
    # with no head axis, only "works" by accident when they're equal).
    return (
        encoder_self_attention_mask[:, None, :, :],
        decoder_self_attention_mask[:, None, :, :],
        decoder_cross_attention_mask[:, None, :, :],
    )
