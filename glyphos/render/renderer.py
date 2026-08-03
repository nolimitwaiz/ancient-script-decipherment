"""Text -> pixel strip via PangoCairo (§ Phase 2.1).

The cairo/gi stack lives in the render environment (micromamba,
~/micromamba-envs/glyphos-render), NOT in the uv venv — importing this module
is always safe; the backend is loaded lazily and a missing backend raises
RenderBackendMissing with setup instructions.

Conventions: strips are float32 (window_h, W) in [0, 1], 1.0 = background,
0.0 = ink. Rendering is deterministic; the cache under runs/render_cache is
regenerable and never data of record. Signs with no Unicode codepoint arrive
as <g>GardinerCode</g> markup and render as a placeholder box (their sign
identity is preserved upstream in the record fields).
"""

import re
from pathlib import Path

import numpy as np

from glyphos.render.config import RenderConfig
from glyphos.utils import paths
from glyphos.utils.hashing import config_hash

PLACEHOLDER = "▯"  # ▯ white vertical rectangle
_G_TAG = re.compile(r"<g>[^<]*</g>")

SCRIPT_FONTS = {
    "egyptian": "Noto Sans Egyptian Hieroglyphs",
    "coptic": "Noto Sans Coptic",
    "linear_b": "Noto Sans Linear B",
    "cuneiform": "Noto Sans Cuneiform",
    "meroitic": "Noto Sans Meroitic",
    "default": "Noto Sans",
}

SETUP_HINT = (
    "render backend missing: install the render environment\n"
    "  micromamba create -p ~/micromamba-envs/glyphos-render -c conda-forge \\\n"
    "      python=3.12 pycairo pygobject pango harfbuzz gobject-introspection numpy\n"
    "and run render code with ~/micromamba-envs/glyphos-render/bin/python "
    "(ancient-script Noto fonts go in ~/.local/share/fonts)"
)


class RenderBackendMissing(RuntimeError):
    pass


def _backend():
    try:
        import cairo
        import gi

        gi.require_version("Pango", "1.0")
        gi.require_version("PangoCairo", "1.0")
        from gi.repository import Pango, PangoCairo
    except (ImportError, ValueError) as exc:
        raise RenderBackendMissing(f"{SETUP_HINT}\n(cause: {exc})") from exc
    return cairo, Pango, PangoCairo


def prepare_text(text: str) -> str:
    """Replace non-Unicode sign markup with the placeholder; collapse whitespace."""
    return " ".join(_G_TAG.sub(PLACEHOLDER, text).split())


def render_strip(text: str, cfg: RenderConfig, font_family: str | None = None) -> np.ndarray:
    cairo, Pango, PangoCairo = _backend()
    text = prepare_text(text)
    if not text:
        raise ValueError("cannot render empty text")
    family = font_family or cfg.font_family

    def _layout(ctx):
        layout = PangoCairo.create_layout(ctx)
        PangoCairo.context_set_resolution(layout.get_context(), cfg.dpi)
        layout.set_font_description(Pango.FontDescription(f"{family} {cfg.font_size_pt}"))
        layout.set_auto_dir(True)  # Pango resolves RTL runs (Hebrew) itself
        layout.set_text(text, -1)
        return layout

    # measure pass
    measure = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
    layout = _layout(cairo.Context(measure))
    w, h = layout.get_pixel_size()
    if w == 0 or h == 0:
        raise ValueError(f"text rendered to zero size with font {family!r}: {text[:40]!r}")

    scale = min(1.0, cfg.window_h / h)
    out_w = max(1, int(np.ceil(w * scale)))
    surface = cairo.ImageSurface(cairo.FORMAT_A8, out_w, cfg.window_h)
    ctx = cairo.Context(surface)
    ctx.translate(0, (cfg.window_h - h * scale) / 2)
    ctx.scale(scale, scale)
    PangoCairo.show_layout(ctx, _layout(ctx))
    surface.flush()

    alpha = np.ndarray(
        shape=(cfg.window_h, surface.get_stride()),
        dtype=np.uint8,
        buffer=surface.get_data(),
    )[:, :out_w]
    ink = alpha.astype(np.float32) / 255.0
    return cfg.background - (cfg.background - cfg.foreground) * ink


def render_cache_dir() -> Path:
    return paths.runs_dir() / "render_cache"


def render_strip_cached(text: str, cfg: RenderConfig, font_family: str | None = None) -> np.ndarray:
    """Deterministic cache keyed by (text, config, font) — regenerable, never
    data of record."""
    key = config_hash(
        {"text": text, "cfg": cfg.__dict__, "font": font_family or cfg.font_family}, n=24
    )
    path = render_cache_dir() / f"{key}.npy"
    if path.exists():
        return np.load(path)
    strip = render_strip(text, cfg, font_family)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, strip)
    return strip
