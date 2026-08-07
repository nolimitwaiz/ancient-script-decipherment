"""Phase 4 evaluation layer: metrics, bootstrap CIs, restoration, conformal."""

import numpy as np
import pytest

from glyphos.eval.metrics import score_translations, sentence_chrf
from glyphos.quant.bootstrap import block_bootstrap, paired_block_bootstrap
from glyphos.quant.conformal import calibrate, evaluate_sets
from glyphos.tasks.restoration import (
    CURRICULUM,
    expected_calibration_error,
    frequency_bins,
    make_mask,
    score_restoration,
)

# -- translation metrics ----------------------------------------------------


def test_perfect_translation_scores_max():
    refs = ["der König gab ein Opfer", "die Sonne ging auf"]
    s = score_translations(refs, refs)
    assert s.chrf == pytest.approx(100.0, abs=1e-6)
    assert s.bleu == pytest.approx(100.0, abs=1e-6)
    assert s.n == 2 and "version" in s.chrf_signature and "tok" in s.bleu_signature


def test_worse_translation_scores_lower():
    refs = ["der König gab ein Opfer"] * 2
    good = ["der König gab ein Opfer", "der König gab Opfer"]
    bad = ["völlig anderer text hier", "noch etwas anderes"]
    assert score_translations(good, refs).chrf > score_translations(bad, refs).chrf


def test_metric_length_mismatch_is_loud():
    with pytest.raises(ValueError, match="differ in length"):
        score_translations(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        score_translations([], [])


def test_sentence_chrf_range():
    assert sentence_chrf("abc", "abc") == pytest.approx(100.0, abs=1e-6)
    assert 0 <= sentence_chrf("xyz", "abc") < 50


# -- bootstrap --------------------------------------------------------------


def test_block_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    docs = [f"d{i // 10}" for i in range(200)]
    values = rng.normal(10.0, 1.0, 200)
    ci = block_bootstrap(docs, lambda idx: float(values[list(idx)].mean()), n_boot=200, seed=1)
    assert ci.low < ci.point < ci.high
    assert ci.high - ci.low < 2.0


def test_block_bootstrap_is_deterministic():
    docs = [f"d{i // 5}" for i in range(50)]
    vals = np.arange(50, dtype=float)
    f = lambda idx: float(vals[list(idx)].mean())  # noqa: E731
    a = block_bootstrap(docs, f, n_boot=100, seed=7)
    b = block_bootstrap(docs, f, n_boot=100, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_block_bootstrap_needs_multiple_documents():
    with pytest.raises(ValueError, match=">= 2 documents"):
        block_bootstrap(["d0"] * 10, lambda idx: 1.0, n_boot=10)


def test_paired_bootstrap_detects_real_and_null_differences():
    docs = [f"d{i // 10}" for i in range(200)]
    a = np.full(200, 10.0)
    real = paired_block_bootstrap(
        docs,
        lambda i: float(a[list(i)].mean()),
        lambda i: float(a[list(i)].mean() - 2.0),
        n_boot=200,
        seed=3,
    )
    assert real.delta == pytest.approx(2.0) and real.significant

    rng = np.random.default_rng(0)
    noise_a, noise_b = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
    null = paired_block_bootstrap(
        docs,
        lambda i: float(noise_a[list(i)].mean()),
        lambda i: float(noise_b[list(i)].mean()),
        n_boot=200,
        seed=3,
    )
    assert null.low < 0 < null.high  # CI covers zero


# -- restoration ------------------------------------------------------------


def test_every_curriculum_mode_masks_within_bounds():
    for mode in CURRICULUM:
        plan = make_mask(20, mode, seed=5)
        assert len(plan) >= 1
        assert all(0 <= p < 20 for p in plan.positions)
        assert len(set(plan.positions)) == len(plan.positions)
        assert make_mask(20, mode, seed=5).positions == plan.positions  # deterministic


def test_damage_mask_is_larger_and_clustered():
    single = make_mask(40, "single", seed=1)
    damage = make_mask(40, "damage", seed=1)
    assert len(damage) > len(single)
    assert len(damage) >= int(0.25 * 40) * 0.5


def test_mask_validates_inputs():
    with pytest.raises(ValueError, match="empty"):
        make_mask(0, "single", seed=1)
    with pytest.raises(ValueError, match="mode must be"):
        make_mask(10, "nonsense", seed=1)


def test_restoration_recall_and_bins():
    counts = {"A": 100, "B": 50, "C": 2}
    bins = frequency_bins(counts, n_bins=3)
    assert bins["A"] == "frequent" and bins["C"] == "rare"
    preds = [["A", "B", "C"], ["B", "A", "C"], ["A", "B", "C"]]
    truths = ["A", "B", "C"]
    s = score_restoration(preds, truths, ks=(1, 3), sign_bins=bins)
    assert s.recall_at[1] == pytest.approx(2 / 3)
    assert s.recall_at[3] == pytest.approx(1.0)
    assert s.recall_by_bin["rare"][1] == 0.0  # C never ranked first
    assert s.recall_by_bin["rare"][3] == 1.0


def test_ece_perfect_and_terrible_calibration():
    assert expected_calibration_error([1.0] * 10, [True] * 10) == pytest.approx(0.0)
    assert expected_calibration_error([1.0] * 10, [False] * 10) == pytest.approx(1.0)


# -- conformal --------------------------------------------------------------


def _synthetic_predictions(n, rng, vocab=("A", "B", "C", "D", "E")):
    labels, probs, truths = [], [], []
    for _ in range(n):
        p = rng.dirichlet(np.ones(len(vocab)) * 0.6)
        order = np.argsort(-p)
        ranked = [vocab[i] for i in order]
        ranked_p = [float(p[i]) for i in order]
        truth = vocab[rng.choice(len(vocab), p=p)]  # truth drawn from the model's own beliefs
        labels.append(ranked)
        probs.append(ranked_p)
        truths.append(truth)
    return labels, probs, truths


def test_conformal_hits_target_coverage():
    rng = np.random.default_rng(0)
    cal = _synthetic_predictions(600, rng)
    test = _synthetic_predictions(600, rng)
    calibration = calibrate(*cal, target_coverage=0.90)
    report = evaluate_sets(calibration, *test)
    # The split-conformal guarantee is coverage >= target. Deterministic APS
    # (no randomised tie-breaking) is known to OVER-cover, so we assert the
    # guarantee holds and that sets stay usably small — not equality.
    assert report.empirical_coverage >= 0.90
    assert 1 <= report.mean_set_size <= len(test[0][0])


def test_higher_target_coverage_gives_larger_sets():
    rng = np.random.default_rng(1)
    cal = _synthetic_predictions(400, rng)
    test = _synthetic_predictions(400, rng)
    low = evaluate_sets(calibrate(*cal, target_coverage=0.5), *test)
    high = evaluate_sets(calibrate(*cal, target_coverage=0.95), *test)
    assert high.mean_set_size > low.mean_set_size
    assert high.empirical_coverage > low.empirical_coverage


def test_conformal_validates_inputs():
    with pytest.raises(ValueError, match="target_coverage"):
        calibrate([["A"]], [[1.0]], ["A"], target_coverage=1.5)
    with pytest.raises(ValueError, match="no calibration"):
        calibrate([], [], [])
