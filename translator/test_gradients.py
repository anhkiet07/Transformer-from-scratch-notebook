"""
Finite-difference gradient check for the hand-written NumPy backward passes
in translator/. Runs a tiny Transformer (toy vocab, drop_prob=0 for a
deterministic forward pass) end to end, compares model.backward()'s
analytic gradients against numerical ones for a handful of entries per
parameter tensor, and fails loudly on any mismatch.

Run from the Transformer/ directory:
    python -m translator.test_gradients
"""

import numpy as np

from translator.data import create_masks
from translator.model import Transformer, cross_entropy_loss


def check_gradients():
    np.random.seed(0)

    START, PAD, END = "<START>", "<PAD>", "<END>"
    english_vocab = [START, PAD, END, "a", "b", "c"]
    vietnamese_vocab = [START, PAD, END, "x", "y", "z"]
    english_to_index = {ch: i for i, ch in enumerate(english_vocab)}
    vietnamese_to_index = {ch: i for i, ch in enumerate(vietnamese_vocab)}

    max_sequence_length = 6
    model = Transformer(
        d_model=8,
        ffn_hidden=16,
        num_heads=2,
        drop_prob=0.0,  # deterministic forward pass, required for finite differences
        num_layers=1,
        max_sequence_length=max_sequence_length,
        target_vocab_size=len(vietnamese_vocab),
        source_char_to_index=english_to_index,
        target_char_to_index=vietnamese_to_index,
        START_TOKEN=START,
        END_TOKEN=END,
        PADDING_TOKEN=PAD,
    )

    eng_batch = ("ab", "abc")
    vi_batch = ("xy", "xyz")
    enc_mask, dec_self_mask, dec_cross_mask = create_masks(eng_batch, vi_batch, max_sequence_length)
    pad_index = vietnamese_to_index[PAD]

    def compute_loss():
        logits = model.forward(
            eng_batch, vi_batch, enc_mask, dec_self_mask, dec_cross_mask,
            enc_start_token=False, enc_end_token=False,
            dec_start_token=True, dec_end_token=True,
        )
        labels = model.decoder.sentence_embedding.batch_tokenize(vi_batch, start_token=False, end_token=True)
        return cross_entropy_loss(logits, labels, pad_index)

    _, dlogits = compute_loss()
    model.zero_grad()
    model.backward(dlogits)

    eps = 1e-5
    rng = np.random.RandomState(1)
    max_rel_error = 0.0
    n_mismatches = 0

    for p in model.parameters():
        flat = p.data.reshape(-1)
        grad_flat = p.grad.reshape(-1)
        n_check = min(5, flat.size)
        indices = rng.choice(flat.size, size=n_check, replace=False)
        for i in indices:
            original = flat[i]
            flat[i] = original + eps
            loss_plus, _ = compute_loss()
            flat[i] = original - eps
            loss_minus, _ = compute_loss()
            flat[i] = original

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grad_flat[i]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_error = abs(numeric - analytic) / denom
            max_rel_error = max(max_rel_error, rel_error)

            # Combined absolute+relative tolerance (standard np.isclose-style
            # check): a pure relative-error test is unreliable when both
            # values are tiny (~1e-5 or smaller), since float64 truncation
            # noise in the finite-difference estimate itself is on that
            # order -- only flag it as a real mismatch once the absolute
            # gap exceeds atol too.
            atol, rtol = 1e-4, 1e-2
            if abs(numeric - analytic) > atol + rtol * abs(analytic):
                n_mismatches += 1
                print(
                    f"MISMATCH shape={p.data.shape} idx={i} "
                    f"numeric={numeric:.6f} analytic={analytic:.6f} rel_error={rel_error:.6f}"
                )

    print(f"Checked {sum(min(5, p.data.size) for p in model.parameters())} gradient entries "
          f"across {len(model.parameters())} parameter tensors.")
    print(f"Max relative error: {max_rel_error:.6e}")
    if n_mismatches:
        raise AssertionError(f"{n_mismatches} gradient mismatches found")
    print("Gradient check passed.")


if __name__ == "__main__":
    check_gradients()
