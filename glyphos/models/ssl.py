"""Self-supervised pixel pretraining (§ Phase 3.3) — the from-scratch
substitute for pretrained pixel models.

Masked-window reconstruction: a fraction of windows is replaced by a learned
mask embedding; the encoder must reconstruct the masked pixels. Loss is MSE
on masked positions only. The trained encoder warm-starts the pixel
translation model — allowed, because every parameter was trained in-repo on
the Phase 2 render stream.
"""

import torch
from torch import nn
from torch.nn import functional as F

from glyphos.models.pixel_encoder import WindowEmbedder
from glyphos.models.transformer import Encoder, ModelConfig, SinusoidalPositions


class MaskedWindowModel(nn.Module):
    def __init__(self, cfg: ModelConfig, window_h: int = 24, window_w: int = 24):
        super().__init__()
        self.cfg = cfg
        self.embed = WindowEmbedder(cfg, window_h, window_w)
        self.pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.encoder = Encoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.reconstruct = nn.Linear(cfg.d_model, window_h * window_w)
        self.window_shape = (window_h, window_w)

    def forward(
        self, windows: torch.Tensor, mask_ratio: float = 0.4, generator=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """windows (B,N,H,W) -> (loss, mask). Mask is (B,N) bool of masked slots."""
        b, n, h, w = windows.shape
        x = self.embed(windows)
        mask = torch.rand(b, n, device=windows.device, generator=generator) < mask_ratio
        if not bool(mask.any()):  # guarantee at least one masked slot
            mask[:, 0] = True
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), x)
        enc = self.encoder(self.pos(x))
        pred = self.reconstruct(enc).reshape(b, n, h, w)
        loss = F.mse_loss(pred[mask], windows[mask])
        return loss, mask

    def encoder_state(self) -> dict:
        """Weights for warm-starting the pixel translation model (in-repo only)."""
        return {
            "embed": self.embed.state_dict(),
            "encoder": self.encoder.state_dict(),
        }


def warm_start_from_ssl(seq2seq, ssl_state: dict) -> None:
    """Load SSL-pretrained front end + encoder into a pixel Seq2Seq."""
    seq2seq.src_embed.load_state_dict(ssl_state["embed"])
    seq2seq.encoder.load_state_dict(ssl_state["encoder"])
