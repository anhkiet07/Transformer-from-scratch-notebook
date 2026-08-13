"""Greedy autoregressive decoding for a trained translator.model.Transformer."""

import numpy as np

from translator.data import END_TOKEN, create_masks


def greedy_decode(model, sentence, max_sequence_length, index_to_vietnamese):
    """Translates a single (already-cleaned, lowercased) English sentence."""
    was_training = model.encoder.sentence_embedding.dropout.training
    model.eval()

    eng_sentence = (sentence,)
    vi_sentence = ("",)
    for word_counter in range(max_sequence_length):
        enc_mask, dec_self_mask, dec_cross_mask = create_masks(eng_sentence, vi_sentence, max_sequence_length)
        logits = model.forward(
            eng_sentence,
            vi_sentence,
            enc_mask,
            dec_self_mask,
            dec_cross_mask,
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=False,
        )
        next_token_index = int(np.argmax(logits[0, word_counter]))
        next_token = index_to_vietnamese[next_token_index]
        if next_token == END_TOKEN:
            break
        vi_sentence = (vi_sentence[0] + next_token,)

    if was_training:
        model.train()
    return vi_sentence[0]
