"""Render configuration (Phase 2) — Salesky et al. defaults, everything overridable.

The visrep recipe: 10pt at 120 DPI via PangoCairo, sentence rendered to a
strip, sliced into overlapping fixed windows h=24, w=24, stride 12.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    font_size_pt: int = 10
    dpi: int = 120
    window_h: int = 24
    window_w: int = 24
    stride: int = 12
    direction: str = "ltr"  # ltr | rtl | ttb
    font_family: str = "Noto Sans"
    # per-script font overrides, e.g. {"egyptian": "Noto Sans Egyptian Hieroglyphs"}
    max_windows: int = 512
    background: float = 1.0  # white
    foreground: float = 0.0  # black


@dataclass(frozen=True)
class DegradeConfig:
    """Composable degradation suite (§ Phase 2.2). `damage_level` in [0, 1]
    scales every enabled transform; individual toggles cut transforms out
    entirely for ablations."""

    damage_level: float = 0.0
    erosion: bool = True
    occlusion: bool = True
    noise: bool = True
    contrast: bool = True
    blur: bool = True
    texture: bool = True
    jitter: bool = True

    def __post_init__(self):
        if not 0.0 <= self.damage_level <= 1.0:
            raise ValueError(f"damage_level must be in [0,1], got {self.damage_level}")
