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
