"""Pixel front-end (§ Phase 3.1, Salesky recipe): each 24x24 window ->
Conv2D(3x3, stride 1) -> BatchNorm2d -> ReLU -> flat linear projection to
d_model. Windows come from glyphos.render.windows.slice_windows.
"""

import torch
from torch import nn

from glyphos.models.transformer import ModelConfig


class WindowEmbedder(nn.Module):
    def __init__(self, cfg: ModelConfig, window_h: int = 24, window_w: int = 24):
        super().__init__()
        self.conv = nn.Conv2d(1, cfg.conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(cfg.conv_channels)
        self.act = nn.ReLU()
        self.proj = nn.Linear(cfg.conv_channels * window_h * window_w, cfg.d_model)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, N, H, W) float in [0,1] -> (B, N, d_model)."""
        b, n, h, w = windows.shape
        x = windows.reshape(b * n, 1, h, w)
        x = self.act(self.bn(self.conv(x)))
        x = x.reshape(b, n, -1)
        return self.proj(x)
