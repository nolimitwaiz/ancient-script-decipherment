"""Frequency-rank matching: the null-model decipherment baseline.

Sort both symbol inventories by unigram frequency and pair them rank-for-rank.
Phase 0 uses it to smoke-test the artificial-decipherment protocol end to end;
from Phase 5 on it is the floor every real alignment model must beat.
"""

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


def char_frequencies(texts: Iterable[str], alphabet: Sequence[str]) -> np.ndarray:
    counts = Counter()
    for text in texts:
        counts.update(ch for ch in text if ch in set(alphabet))
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no in-alphabet characters found; wrong alphabet or empty corpus")
    return np.array([counts[sym] / total for sym in alphabet])


def rank_match(
    src_freqs: np.ndarray,
    tgt_freqs: np.ndarray,
    src_symbols: Sequence[str],
    tgt_symbols: Sequence[str],
) -> dict[str, str]:
    if len(src_symbols) != len(tgt_symbols):
        raise ValueError("rank matching requires equal-size symbol inventories")
    src_order = np.argsort(-src_freqs, kind="stable")
    tgt_order = np.argsort(-tgt_freqs, kind="stable")
    return {src_symbols[i]: tgt_symbols[j] for i, j in zip(src_order, tgt_order, strict=True)}


def decode(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def mapping_accuracy(predicted: dict[str, str], true_key: dict[str, str]) -> float:
    if set(predicted) != set(true_key):
        raise ValueError("predicted mapping and key cover different symbol inventories")
    return sum(predicted[s] == true_key[s] for s in true_key) / len(true_key)


def token_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Micro-averaged word-level exact-match accuracy over parallel sentences."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be parallel")
    correct = total = 0
    for pred, ref in zip(predictions, references, strict=True):
        pred_words, ref_words = pred.split(), ref.split()
        if len(pred_words) != len(ref_words):
            raise ValueError("substitution decoding cannot change word counts")
        total += len(ref_words)
        correct += sum(p == r for p, r in zip(pred_words, ref_words, strict=True))
    if total == 0:
        raise ValueError("empty reference corpus")
    return correct / total
