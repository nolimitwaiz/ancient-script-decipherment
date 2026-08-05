"""Count-based LM baselines (§ Phase 3/4 evaluation).

The sealed-LM hypotheses were preregistered as "substantially better than a
unigram baseline" — this module computes that baseline (and a bigram one) so
the claim is testable rather than rhetorical. Fit on train, scored on valid;
frozen test partitions are never touched here.

Cross-entropy is reported in nats/char to match the trained models' loss, plus
bits/char for readability. Add-alpha smoothing keeps unseen characters finite.
"""

import math
from collections import Counter
from collections.abc import Iterable

UNSEEN = "<unseen>"


def unigram_cross_entropy(
    train_texts: Iterable[str], eval_texts: Iterable[str], alpha: float = 1.0
) -> float:
    """Add-alpha unigram character cross-entropy in nats/char."""
    counts = Counter()
    for t in train_texts:
        counts.update(t)
    vocab = len(counts) + 1  # +1 for unseen mass
    total = sum(counts.values()) + alpha * vocab
    total_lp, n = 0.0, 0
    for t in eval_texts:
        for ch in t:
            p = (counts.get(ch, 0) + alpha) / total
            total_lp += math.log(p)
            n += 1
    if n == 0:
        raise ValueError("empty evaluation text")
    return -total_lp / n


def bigram_cross_entropy(
    train_texts: Iterable[str], eval_texts: Iterable[str], alpha: float = 0.5
) -> float:
    """Add-alpha bigram character cross-entropy in nats/char (stronger floor)."""
    bigrams: dict[str, Counter] = {}
    unigrams = Counter()
    for t in train_texts:
        prev = "\x02"
        for ch in t:
            bigrams.setdefault(prev, Counter())[ch] += 1
            unigrams[ch] += 1
            prev = ch
    vocab = len(unigrams) + 1
    total_lp, n = 0.0, 0
    for t in eval_texts:
        prev = "\x02"
        for ch in t:
            ctx = bigrams.get(prev)
            if ctx is None:
                p = 1.0 / vocab
            else:
                p = (ctx.get(ch, 0) + alpha) / (sum(ctx.values()) + alpha * vocab)
            total_lp += math.log(p)
            n += 1
            prev = ch
    if n == 0:
        raise ValueError("empty evaluation text")
    return -total_lp / n


def nats_to_bits(nats: float) -> float:
    return nats / math.log(2)
