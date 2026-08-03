"""Degradation/augmentation suite (§ Phase 2.2) — pure numpy, seeded, composable.

Every transform takes (img, rng, strength) with strength in [0, 1] and
returns a new float32 image in [0, 1] (1 = background, 0 = ink). `apply`
composes the enabled transforms with strength = cfg.damage_level, so one
scalar sweeps the whole suite for robustness curves. Damage-shaped masks
(`blob_mask`) are exported for the Phase 4 restoration curriculum.
"""

import numpy as np

from glyphos.render.config import DegradeConfig
from glyphos.utils.seed import derive_seed


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur with edge padding; k = odd kernel width."""
    if k <= 1:
        return img
    pad = k // 2
    out = img.astype(np.float32)
    for axis in (0, 1):
        padded = np.concatenate(
            [
                np.repeat(np.take(out, [0], axis=axis), pad, axis=axis),
                out,
                np.repeat(np.take(out, [-1], axis=axis), pad, axis=axis),
            ],
            axis=axis,
        )
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
        lead = np.take(csum, range(k - 1, padded.shape[axis]), axis=axis)
        lag = np.concatenate(
            [
                np.zeros_like(np.take(csum, [0], axis=axis)),
                np.take(csum, range(0, padded.shape[axis] - k), axis=axis),
            ],
            axis=axis,
        )
        out = ((lead - lag) / k).astype(np.float32)
    return out


def blob_mask(shape: tuple[int, int], rng: np.random.Generator, coverage: float) -> np.ndarray:
    """Irregular damage-shaped boolean mask covering ~`coverage` of the image:
    thresholded smoothed noise gives realistic blob/edge-loss geometry."""
    if coverage <= 0:
        return np.zeros(shape, dtype=bool)
    noise = rng.random(shape).astype(np.float32)
    smooth = _box_blur(noise, k=max(3, min(shape) // 3 | 1))
    threshold = np.quantile(smooth, coverage)
    return smooth <= threshold


def erode(img, rng, s):
    """Stroke erosion/thinning: probabilistic 3x3 background-dilation on ink."""
    if s <= 0:
        return img
    shifted = np.stack(
        [np.roll(np.roll(img, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    )
    dilated = shifted.max(axis=0)  # background is high -> max eats ink edges
    where = rng.random(img.shape) < (0.6 * s)
    return np.where(where, dilated, img)


def occlude(img, rng, s):
    """Damage-shaped occlusion patches reset to background."""
    mask = blob_mask(img.shape, rng, coverage=0.25 * s)
    return np.where(mask, 1.0, img)


def add_noise(img, rng, s):
    out = img + rng.normal(0.0, 0.12 * s, img.shape).astype(np.float32)
    salt = rng.random(img.shape) < 0.03 * s
    pepper = rng.random(img.shape) < 0.03 * s
    out = np.where(salt, 1.0, out)
    out = np.where(pepper, 0.0, out)
    return out


def shift_contrast(img, rng, s):
    gain = 1.0 - 0.5 * s * rng.random()
    bias = (rng.random() - 0.5) * 0.3 * s
    return (img - 0.5) * gain + 0.5 + bias


def blur(img, rng, s):
    k = 1 + 2 * round(1.5 * s * rng.random() + 0.5 * s)
    return _box_blur(img, k)


def background_texture(img, rng, s):
    """Procedural stone/papyrus mottling multiplied into the background."""
    if s <= 0:
        return img
    coarse = rng.random((max(2, -(-img.shape[0] // 6)), max(2, -(-img.shape[1] // 6))))
    field = np.kron(coarse, np.ones((6, 6)))[: img.shape[0], : img.shape[1]]
    field = _box_blur(field.astype(np.float32), 5)
    texture = 1.0 - 0.45 * s * field
    return img * texture


def jitter(img, rng, s):
    """Slight rotation/scale via inverse-mapped nearest-neighbor affine."""
    if s <= 0:
        return img
    angle = np.deg2rad(rng.uniform(-3.0, 3.0) * s)
    scale = 1.0 + rng.uniform(-0.05, 0.05) * s
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices(img.shape).astype(np.float32)
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    src_y = cos * (yy - cy) + sin * (xx - cx) + cy
    src_x = -sin * (yy - cy) + cos * (xx - cx) + cx
    src_y = np.clip(np.rint(src_y), 0, h - 1).astype(np.int64)
    src_x = np.clip(np.rint(src_x), 0, w - 1).astype(np.int64)
    return img[src_y, src_x]


_PIPELINE = (
    ("jitter", jitter),
    ("erosion", erode),
    ("texture", background_texture),
    ("occlusion", occlude),
    ("blur", blur),
    ("contrast", shift_contrast),
    ("noise", add_noise),
)


def apply(img: np.ndarray, cfg: DegradeConfig, seed: int) -> np.ndarray:
    """Apply the enabled transforms at strength = damage_level; deterministic
    in (img, cfg, seed)."""
    out = img.astype(np.float32)
    if cfg.damage_level <= 0:
        return out
    for name, fn in _PIPELINE:
        if getattr(cfg, name):
            rng = np.random.default_rng(derive_seed(seed, "degrade", name))
            out = fn(out, rng, cfg.damage_level)
    return np.clip(out, 0.0, 1.0)
