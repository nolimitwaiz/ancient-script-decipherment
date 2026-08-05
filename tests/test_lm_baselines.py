import math

import pytest

from glyphos.eval.lm_baselines import (
    bigram_cross_entropy,
    nats_to_bits,
    unigram_cross_entropy,
)


def test_uniform_text_matches_analytic_entropy():
    """Four equiprobable characters -> ~ln(4) nats/char (plus smoothing slack)."""
    train = ["abcd" * 500]
    ce = unigram_cross_entropy(train, ["abcd" * 10])
    assert abs(ce - math.log(4)) < 0.02


def test_bigram_beats_unigram_on_structured_text():
    train = ["abababab" * 100]
    uni = unigram_cross_entropy(train, ["abababab" * 5])
    bi = bigram_cross_entropy(train, ["abababab" * 5])
    assert bi < uni  # perfectly predictable next char


def test_unseen_characters_stay_finite():
    ce = unigram_cross_entropy(["aaaa"], ["zzzz"])
    assert math.isfinite(ce) and ce > 0


def test_nats_to_bits():
    assert nats_to_bits(math.log(2)) == pytest.approx(1.0)


def test_empty_eval_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        unigram_cross_entropy(["abc"], [])
