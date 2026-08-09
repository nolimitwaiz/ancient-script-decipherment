"""Load REAL artifact images into the same strip format the renderer emits.

Everything downstream (window slicing, degradation, the pixel encoder) is
shared with rendered text, so a real lineart and a font render are
interchangeable inputs — which is exactly what makes the comparison between
them meaningful.

Contract matches `render_strip`: float32 (window_h, W) in [0, 1], 1.0 =
background, 0.0 = ink. Source images are polarity-corrected (scanned linearts
vary), TRIMMED to their ink bounding box, height-normalised to window_h, and
width-capped to the window budget.

Trimming is not cosmetic. LogogramNLP textlines are padded onto a fixed
8464px canvas while the glyphs occupy only the first 1.7-3.6% of it; without
trimming ~98% of the sliced windows would be blank padding and the encoder
would train on almost nothing.
"""

from pathlib import Path

import numpy as np

from glyphos.render.config import RenderConfig
from glyphos.render.renderer import max_strip_width


class ImageBackendMissing(RuntimeError):
    pass


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImageBackendMissing(
            "Pillow is required to load artifact images: uv add pillow"
        ) from exc
    return Image


def trim_to_ink(arr: np.ndarray, threshold: float = 0.5, margin: int = 2) -> np.ndarray:
    """Crop away background padding around the inked region.

    Returns the input unchanged if it contains no ink (caller decides whether
    a blank image is an error) — never returns an empty array.
    """
    ink = arr < threshold
    if not ink.any():
        return arr
    cols = ink.any(axis=0).nonzero()[0]
    rows = ink.any(axis=1).nonzero()[0]
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(arr.shape[1], int(cols.max()) + 1 + margin)
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(arr.shape[0], int(rows.max()) + 1 + margin)
    return arr[r0:r1, c0:c1]


def load_strip(
    path: str | Path, cfg: RenderConfig, invert: str = "auto", trim: bool = True
) -> np.ndarray:
    """Load an image as a strip.

    `invert="auto"` decides polarity from the median: scholarly linearts are
    dark-on-light, photographs of incised stone are often the reverse, and
    getting this wrong silently feeds the encoder a negative.
    """
    Image = _pil()
    img = Image.open(path).convert("L")
    if img.size[0] == 0 or img.size[1] == 0:
        raise ValueError(f"{path}: empty image")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert == "always" or (invert == "auto" and float(np.median(arr)) < 0.5):
        arr = 1.0 - arr
    if trim:
        arr = trim_to_ink(arr)

    h, w = arr.shape
    scale = cfg.window_h / h
    new_w = max(1, min(round(w * scale), max_strip_width(cfg)))
    resized = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize(
        (new_w, cfg.window_h), Image.LANCZOS
    )
    return np.ascontiguousarray(np.asarray(resized, dtype=np.float32) / 255.0)


def ink_fraction(strip: np.ndarray, threshold: float = 0.5) -> float:
    """Share of non-background pixels — a cheap sanity check that a load
    produced glyphs rather than a blank or fully-inked frame."""
    return float((strip < threshold).mean())
