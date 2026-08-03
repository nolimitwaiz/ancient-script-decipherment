#!/usr/bin/env python
"""Visual regression grids (§ Phase 2.2 deliverable): per script, render a
sample sentence clean, at four damage levels, and under each single transform,
into one labeled PNG under docs/render_samples/.

Run with the render environment:
    PYTHONPATH=. ~/micromamba-envs/glyphos-render/bin/python scripts/render_samples.py
"""

import numpy as np

from glyphos.render import degrade
from glyphos.render.config import DegradeConfig, RenderConfig
from glyphos.render.renderer import SCRIPT_FONTS, render_strip
from glyphos.utils import paths

SAMPLES = {
    "egyptian": ("𓇋𓅱 𓅯𓄿 𓇯𓅺𓏏𓏭𓅂𓀜𓀀𓏥 <g>Ff101</g> 𓈝𓅓𓏏𓂻 𓂋 𓌸𓂋𓇋𓇋𓏏𓈇𓏤", "egyptian"),
    "coptic": ("ⲡⲃⲓⲟⲥ ⲁⲩⲱ ⲧⲡⲟⲗⲓⲧⲉⲓⲁ ⲙⲡⲉⲛⲡⲉⲧⲟⲩⲁⲁⲃ ⲛⲉⲓⲱⲧ", "coptic"),
    "hebrew": ("בְּ רֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם", "default"),
    "greek": ("ἐν ἀρχῇ ἦν ὁ λόγος καὶ ὁ λόγος ἦν πρὸς τὸν θεόν", "default"),
    "ugaritic_translit": ("ab ib awyb bd sdq yrgm lb", "default"),
    "linear_b": ("𐀀𐀁𐀪𐀦𐀲 𐀀𐀁𐀴𐀵", "linear_b"),
    "cuneiform": ("𒀭𒂗𒆤 𒈗 𒆠𒂗𒄀", "cuneiform"),
    "meroitic": ("𐦠𐦡𐦢𐦣𐦤𐦥𐦦𐦧", "meroitic"),
}

DAMAGE_SWEEP = (0.25, 0.5, 0.75, 1.0)
SOLO_STRENGTH = 0.7
SOLO_TRANSFORMS = ("erosion", "occlusion", "noise", "contrast", "blur", "texture", "jitter")
SEED = 1234
PAD = 4


def solo_config(name: str, level: float) -> DegradeConfig:
    flags = {t: (t == name) for t in SOLO_TRANSFORMS}
    return DegradeConfig(damage_level=level, **flags)


def save_png(gray: np.ndarray, path) -> None:
    import cairo

    h, w = gray.shape
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    buf = np.ndarray(
        shape=(h, surface.get_stride() // 4, 4), dtype=np.uint8, buffer=surface.get_data()
    )
    v = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    for c in range(3):
        buf[:, :w, c] = v
    surface.mark_dirty()
    surface.write_to_png(str(path))


def grid(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    canvas = []
    for r in rows:
        padded = np.ones((r.shape[0] + PAD, width), dtype=np.float32) * 0.85
        padded[: r.shape[0], : r.shape[1]] = r
        canvas.append(padded)
    return np.concatenate(canvas, axis=0)


def main() -> int:
    out_dir = paths.repo_root() / "docs" / "render_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RenderConfig()
    for name, (text, script) in SAMPLES.items():
        font = SCRIPT_FONTS[script]
        clean = render_strip(text, cfg, font)
        rows = [clean]
        rows += [
            degrade.apply(clean, DegradeConfig(damage_level=lvl), SEED) for lvl in DAMAGE_SWEEP
        ]
        rows += [degrade.apply(clean, solo_config(t, SOLO_STRENGTH), SEED) for t in SOLO_TRANSFORMS]
        path = out_dir / f"{name}.png"
        save_png(grid(rows), path)
        print(
            f"[samples] {name}: {clean.shape[1]}px strip -> {path.name} "
            f"(clean + {len(DAMAGE_SWEEP)} damage levels + {len(SOLO_TRANSFORMS)} solo)"
        )
    print(f"[samples] wrote {len(SAMPLES)} grids to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
