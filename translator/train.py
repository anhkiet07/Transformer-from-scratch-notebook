"""
Trains the pure-NumPy Transformer (translator/model.py) on a small
English -> Vietnamese character-level dataset built from data/train.en and
data/train.vi, and saves the result to translator/checkpoint.pkl.

Run from the Transformer/ directory:
    python -m translator.train
"""

import os
import time

from translator.checkpoint import save_checkpoint
from translator.data import END_TOKEN, PADDING_TOKEN, START_TOKEN, create_masks, iterate_batches, load_dataset
from translator.model import Transformer, cross_entropy_loss
from translator.optimizer import Adam
from translator.translate import greedy_decode

# Small on purpose: this trains with hand-written NumPy backprop on CPU,
# no GPU/vectorized-autograd acceleration, so the whole setup is sized to
# actually finish (see docs/known-issues.md #4 for why src/ can't train at
# all -- it has no backward pass).
D_MODEL = 64
NUM_HEADS = 4
FFN_HIDDEN = 128
NUM_LAYERS = 1
DROP_PROB = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 3e-4
NUM_SENTENCES = 300
MAX_SENTENCE_LENGTH = 30

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pkl")


def main():
    print("Loading data...")
    english_sentences, vietnamese_sentences, max_sequence_length, english_vocab, vietnamese_vocab = load_dataset(
        num_sentences=NUM_SENTENCES, max_sentence_length=MAX_SENTENCE_LENGTH
    )
    _, _, english_char_to_index = english_vocab
    vietnamese_vocabulary, index_to_vietnamese, vietnamese_char_to_index = vietnamese_vocab
    vi_vocab_size = len(vietnamese_vocabulary)

    print(f"Sentence pairs: {len(english_sentences)}")
    print(f"English vocab size: {len(english_char_to_index)}")
    print(f"Vietnamese vocab size: {vi_vocab_size}")
    print(f"max_sequence_length: {max_sequence_length}")

    model = Transformer(
        D_MODEL, FFN_HIDDEN, NUM_HEADS, DROP_PROB, NUM_LAYERS,
        max_sequence_length, vi_vocab_size,
        english_char_to_index, vietnamese_char_to_index,
        START_TOKEN, END_TOKEN, PADDING_TOKEN,
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    pad_index = vietnamese_char_to_index[PADDING_TOKEN]

    print(f"Total parameters: {sum(p.data.size for p in model.parameters())}")

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        for eng_batch, vi_batch in iterate_batches(english_sentences, vietnamese_sentences, BATCH_SIZE, seed=epoch):
            enc_mask, dec_self_mask, dec_cross_mask = create_masks(eng_batch, vi_batch, max_sequence_length)
            logits = model.forward(
                eng_batch, vi_batch, enc_mask, dec_self_mask, dec_cross_mask,
                enc_start_token=False, enc_end_token=False,
                dec_start_token=True, dec_end_token=True,
            )
            labels = model.decoder.sentence_embedding.batch_tokenize(vi_batch, start_token=False, end_token=True)
            loss, dlogits = cross_entropy_loss(logits, labels, pad_index)

            model.zero_grad()
            model.backward(dlogits)
            optimizer.step()

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
            sample_en = english_sentences[0]
            sample_vi = vietnamese_sentences[0]
            prediction = greedy_decode(model, sample_en, max_sequence_length, index_to_vietnamese)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  loss={avg_loss:.4f}  ({elapsed:.1f}s elapsed)")
            print(f"  EN: {sample_en}")
            print(f"  VI (target):     {sample_vi}")
            print(f"  VI (prediction): {prediction}")
        else:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  loss={avg_loss:.4f}  ({elapsed:.1f}s elapsed)")

    hyperparams = dict(
        d_model=D_MODEL, ffn_hidden=FFN_HIDDEN, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, drop_prob=DROP_PROB, max_sequence_length=max_sequence_length,
    )
    save_checkpoint(CHECKPOINT_PATH, model, hyperparams, english_vocab, vietnamese_vocab)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
