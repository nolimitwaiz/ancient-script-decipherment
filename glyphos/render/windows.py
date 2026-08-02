"""Strip -> overlapping fixed windows (§ Phase 2.1).

Pure numpy; no rendering backend needed. A strip is a float32 array (H, W) in
[0, 1]; windows are (N, h, w) slices at the configured stride, right-padded
with background so the final partial window is kept (never silently dropped).
"""

import numpy as np

from glyphos.render.config import RenderConfig


def slice_windows(strip: np.ndarray, cfg: RenderConfig) -> np.ndarray:
    if strip.ndim != 2:
        raise ValueError(f"strip must be 2-D (H, W), got shape {strip.shape}")
    h, w = strip.shape
    if h != cfg.window_h:
        raise ValueError(f"strip height {h} != window_h {cfg.window_h}")
    if w == 0:
        raise ValueError("empty strip")

    n = max(1, -(-max(w - cfg.window_w, 0) // cfg.stride) + 1)
    n = min(n, cfg.max_windows)
    padded_w = (n - 1) * cfg.stride + cfg.window_w
    padded = np.full((h, padded_w), cfg.background, dtype=np.float32)
    padded[:, : min(w, padded_w)] = strip[:, : min(w, padded_w)]

    return np.stack([padded[:, i * cfg.stride : i * cfg.stride + cfg.window_w] for i in range(n)])
