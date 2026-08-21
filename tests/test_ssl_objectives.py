"""SSL objectives: masking geometry (the leak fix) and I-JEPA collapse safety."""

from itertools import pairwise

import pytest
import torch

from glyphos.models.seq2seq import Seq2Seq
from glyphos.models.ssl import (
    JEPAModel,
    MaskedWindowModel,
    RepresentationCollapse,
    make_mask,
    random_mask,
    span_mask,
    warm_start_from_ssl,
)
from glyphos.models.transformer import ModelConfig

TINY = ModelConfig(
    d_model=32,
    n_heads=2,
    ff=64,
    enc_layers=2,
    dec_layers=1,
    max_len=64,
    vocab_size=50,
    conv_channels=4,
)


def _neighbour_intact_fraction(mask: torch.Tensor) -> float:
    """Fraction of masked windows keeping an unmasked immediate neighbour.

    With 50%-overlapping windows such a window is reconstructable by copying —
    this is exactly the leak that made random masking trivial (run 1745681).
    """
    b, n = mask.shape
    leaky = total = 0
    for i in range(b):
        for j in range(n):
            if not mask[i, j]:
                continue
            total += 1
            left_ok = j > 0 and not mask[i, j - 1]
            right_ok = j < n - 1 and not mask[i, j + 1]
            leaky += int(left_ok or right_ok)
    return leaky / max(total, 1)


def test_span_masking_closes_the_overlap_leak():
    torch.manual_seed(0)
    rnd = random_mask(8, 64, 0.4, device="cpu")
    spn = span_mask(8, 64, 0.4, device="cpu")
    leak_random = _neighbour_intact_fraction(rnd)
    leak_span = _neighbour_intact_fraction(spn)
    assert leak_random > 0.7, f"expected random masking to leak badly, got {leak_random:.2f}"
    assert leak_span < 0.4, f"span masking should mostly hide interiors, got {leak_span:.2f}"
    assert leak_span < leak_random / 2


def test_span_mask_hits_target_ratio_and_is_contiguous():
    m = span_mask(4, 60, 0.4, device="cpu")
    frac = m.float().mean().item()
    assert 0.2 <= frac <= 0.75
    for row in m:
        idx = row.nonzero().flatten().tolist()
        runs = 1 + sum(1 for a, b in pairwise(idx) if b != a + 1)
        assert runs <= max(2, len(idx) // 3)  # few, long runs — not scattered singles


def test_masks_never_empty_and_mode_validated():
    assert bool(span_mask(2, 5, 0.01, device="cpu").any())
    assert bool(random_mask(2, 5, 0.0, device="cpu").any())
    with pytest.raises(ValueError, match="mask_mode"):
        make_mask("bogus", 2, 8, 0.4, device="cpu")


def test_mae_defaults_to_span_masking_and_trains():
    torch.manual_seed(0)
    model = MaskedWindowModel(TINY)
    assert model.mask_mode == "span"
    loss, mask = model(torch.rand(2, 12, 24, 24))
    assert loss.requires_grad and bool(mask.any())
    loss.backward()


def test_jepa_forward_and_ema_target_is_frozen():
    torch.manual_seed(0)
    model = JEPAModel(TINY, warmup_steps=10**6)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    loss, stats = model(torch.rand(2, 12, 24, 24))
    assert loss.requires_grad
    assert stats["target_var"] >= 0 and 0 < stats["mask_frac"] < 1
    loss.backward()
    assert model.encoder.layers[0].attn.in_proj_weight.grad is not None


def test_jepa_ema_moves_target_toward_online_encoder():
    torch.manual_seed(0)
    model = JEPAModel(TINY, ema_momentum=0.5, warmup_steps=10**6).train()
    before = next(model.target_encoder.parameters()).clone()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    model(torch.rand(2, 10, 24, 24))
    after = next(model.target_encoder.parameters())
    assert not torch.allclose(before, after), "EMA target never updated"


def test_jepa_raises_on_representation_collapse():
    """A collapsed encoder posts a beautiful loss and carries zero information —
    the detector is what makes this arm falsifiable."""
    torch.manual_seed(0)
    model = JEPAModel(TINY, warmup_steps=1, collapse_ratio=0.5).train()
    model(torch.rand(2, 12, 24, 24))  # step 1 sets the reference variance
    with torch.no_grad():  # simulate collapse: target encoder outputs a constant
        for p in model.target_encoder.parameters():
            p.zero_()
        for p in model.target_embed.parameters():
            p.zero_()
    model.ema_momentum = 1.0  # freeze the collapsed state
    with pytest.raises(RepresentationCollapse, match="collapsed"):
        for _ in range(3):
            model(torch.rand(2, 12, 24, 24))


def test_jepa_encoder_warm_starts_a_pixel_model():
    torch.manual_seed(0)
    jepa = JEPAModel(TINY, warmup_steps=10**6)
    pix = Seq2Seq(TINY, frontend="pixel")
    warm_start_from_ssl(pix, jepa.encoder_state())
    for a, b in zip(pix.encoder.parameters(), jepa.encoder.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_normalised_targets_make_the_loss_interpretable():
    """Rendered strips are ~95% background; unnormalised MSE is dominated by
    empty space (real TLA renders: all-white scores 0.0344, per-window mean
    0.0289 — both BETTER than the first runs' 0.042). Per-window
    standardisation fixes the scale: predicting the mean scores ~1.0."""
    from glyphos.models.ssl import normalize_windows

    torch.manual_seed(0)
    sparse = torch.ones(4, 8, 24, 24)
    sparse[:, :, 10:13, 10:13] = 0.0  # ~1.5% ink, like a real glyph strip

    raw_trivial = ((sparse - sparse.mean(dim=(-2, -1), keepdim=True)) ** 2).mean()
    assert raw_trivial < 0.05, "raw MSE is background-dominated (that is the bug)"

    normed = normalize_windows(sparse)
    norm_trivial = ((normed - normed.mean(dim=(-2, -1), keepdim=True)) ** 2).mean()
    assert 0.9 <= float(norm_trivial) <= 1.1, "normalised: predicting the mean must score ~1.0"


def test_mae_uses_normalised_targets_by_default():
    torch.manual_seed(0)
    model = MaskedWindowModel(TINY)
    assert model.normalize_targets
    sparse = torch.ones(2, 10, 24, 24)
    sparse[:, :, 8:12, 8:12] = 0.0
    loss, _ = model(sparse)
    # an untrained model on standardised targets sits near or above 1.0;
    # if it were reading raw pixels it would start near 0.03 and look "great"
    assert float(loss) > 0.3, f"loss {float(loss):.4f} suspiciously low — targets unnormalised?"


def test_collapse_detector_arms_within_a_kill_gate():
    """Kill-gates are 200 steps; a 500-step warmup meant the detector never
    armed in the very runs meant to catch problems."""
    assert JEPAModel(TINY).warmup_steps <= 200


def test_jepa_reports_its_trivial_baseline():
    """Same discipline as the MAE arm: a latent-prediction loss is meaningless
    without the score of predicting the mean target."""
    torch.manual_seed(0)
    model = JEPAModel(TINY, warmup_steps=10**6)
    _, stats = model(torch.rand(2, 12, 24, 24))
    assert stats["trivial_baseline"] > 0
    assert "loss_over_trivial" in stats
