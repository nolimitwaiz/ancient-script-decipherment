"""Render-engine tests. Window slicing and the degradation suite are pure
numpy and always run; renderer tests need the cairo backend and skip cleanly
in environments without it (run the full suite under the render env too:
~/micromamba-envs/glyphos-render/bin/python -m pytest)."""

import numpy as np
import pytest

from glyphos.render import degrade
from glyphos.render.config import DegradeConfig, RenderConfig
from glyphos.render.renderer import prepare_text
from glyphos.render.windows import slice_windows

CFG = RenderConfig()


def _strip(w: int = 100) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((CFG.window_h, w)).astype(np.float32)


# -- windows ----------------------------------------------------------------


def test_slice_windows_counts_and_overlap():
    strip = _strip(100)
    wins = slice_windows(strip, CFG)
    # ceil((100-24)/12)+1 = 8 windows
    assert wins.shape == (8, 24, 24)
    np.testing.assert_array_equal(wins[0], strip[:, 0:24])
    np.testing.assert_array_equal(wins[1][:, :12], strip[:, 12:24])  # overlap region


def test_slice_windows_pads_final_window_with_background():
    strip = _strip(30)
    wins = slice_windows(strip, CFG)
    assert wins.shape == (2, 24, 24)
    assert (wins[1][:, 18:] == CFG.background).all()  # 12+24=36 > 30 -> padded


def test_slice_windows_narrow_strip():
    wins = slice_windows(_strip(10), CFG)
    assert wins.shape == (1, 24, 24)


def test_slice_windows_validates():
    with pytest.raises(ValueError, match="window_h"):
        slice_windows(np.zeros((10, 50), dtype=np.float32), CFG)
    with pytest.raises(ValueError, match="2-D"):
        slice_windows(np.zeros((24, 50, 1), dtype=np.float32), CFG)


def test_slice_windows_caps_at_max_windows():
    cfg = RenderConfig(max_windows=3)
    assert slice_windows(_strip(1000), cfg).shape == (3, 24, 24)


# -- degradations -----------------------------------------------------------


def test_damage_zero_is_identity():
    img = _strip(80)
    out = degrade.apply(img, DegradeConfig(damage_level=0.0), seed=7)
    np.testing.assert_array_equal(out, img)


def test_damage_is_deterministic_and_seed_sensitive():
    img = _strip(80)
    cfg = DegradeConfig(damage_level=0.6)
    a = degrade.apply(img, cfg, seed=7)
    b = degrade.apply(img, cfg, seed=7)
    c = degrade.apply(img, cfg, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_damage_scales_with_level():
    img = _strip(80)
    delta = [
        np.abs(degrade.apply(img, DegradeConfig(damage_level=lvl), seed=7) - img).mean()
        for lvl in (0.2, 0.9)
    ]
    assert delta[1] > delta[0] > 0


def test_damage_preserves_shape_and_range():
    img = _strip(80)
    out = degrade.apply(img, DegradeConfig(damage_level=1.0), seed=7)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_transform_toggles_cut_transforms():
    img = _strip(80)
    only_noise = DegradeConfig(
        damage_level=0.8,
        erosion=False,
        occlusion=False,
        contrast=False,
        blur=False,
        texture=False,
        jitter=False,
    )
    out = degrade.apply(img, only_noise, seed=7)
    assert not np.array_equal(out, img)


def test_blob_mask_coverage_grows():
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
    small = degrade.blob_mask((24, 200), rng1, 0.1).mean()
    large = degrade.blob_mask((24, 200), rng2, 0.4).mean()
    assert large > small > 0
    assert degrade.blob_mask((24, 200), rng1, 0.0).sum() == 0


def test_degrade_config_validates_level():
    with pytest.raises(ValueError, match="damage_level"):
        DegradeConfig(damage_level=1.5)


# -- renderer (needs cairo backend; skips cleanly without it) ---------------


def test_prepare_text_replaces_sign_markup():
    assert prepare_text("𓇋𓅱 <g>Ff101</g>  𓏤") == "𓇋𓅱 ▯ 𓏤"


@pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("cairo"),
    reason="cairo backend not in this environment",
)
def test_render_strip_produces_ink():
    from glyphos.render.renderer import render_strip

    strip = render_strip("ἐν ἀρχῇ ἦν ὁ λόγος", CFG)
    assert strip.shape[0] == CFG.window_h
    assert strip.min() < 0.5  # ink present
    assert strip.max() > 0.9  # background present
    again = render_strip("ἐν ἀρχῇ ἦν ὁ λόγος", CFG)
    np.testing.assert_array_equal(strip, again)


@pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("cairo"),
    reason="cairo backend not in this environment",
)
def test_render_strip_truncates_beyond_window_budget():
    """A Greek-paragraph-sized input must fit the window budget, not explode
    cairo (regression: SSL kill-gate 1709839)."""
    from glyphos.render.renderer import max_strip_width, render_strip

    limit = max_strip_width(CFG)
    strip = render_strip("λόγος " * 4000, CFG)
    assert strip.shape[0] == CFG.window_h
    assert limit * 0.9 <= strip.shape[1] <= limit  # fills the budget, never exceeds it
    tiny = RenderConfig(max_windows=4)
    assert render_strip("λόγος " * 4000, tiny).shape[1] <= max_strip_width(tiny)


def test_trim_to_ink_removes_padding():
    """LogogramNLP textlines are padded onto an 8464px canvas while glyphs
    occupy the first ~2%; untrimmed, ~98% of sliced windows are blank."""
    from glyphos.render.images import trim_to_ink

    padded = np.ones((16, 4000), dtype=np.float32)
    padded[4:12, 10:210] = 0.0  # the only ink
    trimmed = trim_to_ink(padded)
    assert trimmed.shape[1] < 250, f"padding survived: width {trimmed.shape[1]}"
    assert trimmed.shape[0] <= 16
    assert float((trimmed < 0.5).mean()) > float((padded < 0.5).mean()) * 10


def test_trim_to_ink_passes_through_blank_images():
    from glyphos.render.images import trim_to_ink

    blank = np.ones((16, 100), dtype=np.float32)
    assert trim_to_ink(blank).shape == blank.shape  # never returns an empty array


@pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("PIL"),
    reason="Pillow not installed",
)
def test_load_strip_matches_render_contract(tmp_path):
    """A real image and a font render must be interchangeable downstream."""
    from PIL import Image

    from glyphos.render.images import load_strip

    arr = np.full((16, 800), 255, dtype=np.uint8)
    arr[3:13, 5:120] = 0
    Image.fromarray(arr).save(tmp_path / "line.png")
    strip = load_strip(tmp_path / "line.png", CFG)
    assert strip.shape[0] == CFG.window_h
    assert strip.dtype == np.float32
    assert strip.min() >= 0.0 and strip.max() <= 1.0
    assert strip.shape[1] <= 400  # trimmed, not the full 800-wide canvas
    slice_windows(strip, CFG)  # must feed the standard window path
