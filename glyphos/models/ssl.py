"""Self-supervised pixel pretraining (§ Phase 3.3) — the from-scratch
substitute for pretrained pixel models.

Two objectives, deliberately separable so we can tell WHICH change helps:

- `MaskedWindowModel` (MAE family): reconstruct masked window pixels.
- `JEPAModel` (I-JEPA family): predict the *representations* of masked target
  spans, produced by an EMA target encoder. No pixel reconstruction.

CRITICAL — masking geometry. Windows are 24px wide at stride 12, so
consecutive windows overlap by 50%. Under independent random masking a masked
window is recoverable by copying from its neighbours (~84% of masked windows
retain an intact neighbour), which makes reconstruction nearly trivial —
observed as training loss 0.005 in run 1745681. `mask_mode="span"` masks
contiguous runs instead, so a masked span's interior is not visible in any
surviving window. JEPA uses span masking by construction.

CRITICAL — target normalisation. Rendered strips are ~95% background, so raw
pixel MSE is dominated by empty space: predicting all-white scores 0.0344 and
predicting the per-window mean scores 0.0289 on real TLA renders, while the
first unnormalised runs sat at 0.042 — i.e. WORSE than trivial. Targets are
therefore per-window standardised (the MAE paper's `norm_pix_loss`), which
makes the loss interpretable: predicting the window mean scores ~1.0, so any
loss below 1.0 is genuine structure and the number is comparable across runs.

Both self-distillation-style objectives can COLLAPSE (constant output, perfect
loss, zero information). `JEPAModel` therefore tracks target-representation
variance and raises `RepresentationCollapse` rather than letting a dead run
look like a converged one.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from glyphos.models.pixel_encoder import WindowEmbedder
from glyphos.models.transformer import Encoder, ModelConfig, SinusoidalPositions


class RepresentationCollapse(RuntimeError):
    """Raised when an encoder's outputs lose variance — a dead run."""


def random_mask(b: int, n: int, ratio: float, device, generator=None) -> torch.Tensor:
    mask = torch.rand(b, n, device=device, generator=generator) < ratio
    if not bool(mask.any()):
        mask[:, 0] = True
    return mask


def span_mask(
    b: int,
    n: int,
    ratio: float,
    device,
    generator=None,
    min_span: int = 4,
) -> torch.Tensor:
    """Mask contiguous runs of windows.

    With 50%-overlapping windows, only spans longer than one window hide
    information: the interior of a masked span appears in no surviving window.
    `min_span` >= 3 guarantees at least one fully-hidden interior window.
    """
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    target = max(min_span, round(ratio * n))
    for i in range(b):
        filled = 0
        guard = 0
        while filled < target and guard < 4 * n:
            guard += 1
            length = int(
                torch.randint(min_span, max(min_span + 1, target + 1), (1,), generator=generator)
            )
            length = min(length, n)
            start = int(torch.randint(0, max(1, n - length + 1), (1,), generator=generator))
            before = int(mask[i].sum())
            mask[i, start : start + length] = True
            filled += int(mask[i].sum()) - before
        if not bool(mask[i].any()):
            mask[i, : min(min_span, n)] = True
    return mask


def make_mask(mode: str, b: int, n: int, ratio: float, device, generator=None) -> torch.Tensor:
    if mode == "random":
        return random_mask(b, n, ratio, device, generator)
    if mode == "span":
        return span_mask(b, n, ratio, device, generator)
    raise ValueError(f"mask_mode must be 'random' or 'span', got {mode!r}")


def normalize_windows(windows: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-window standardisation of reconstruction targets (MAE norm_pix_loss).

    Removes the background's dominance: after this, predicting a window's mean
    scores ~1.0, so the loss reads directly as "fraction of within-window
    variance left unexplained".
    """
    mean = windows.mean(dim=(-2, -1), keepdim=True)
    std = windows.std(dim=(-2, -1), keepdim=True)
    return (windows - mean) / (std + eps)


class MaskedWindowModel(nn.Module):
    """MAE-family: reconstruct masked window pixels (per-window standardised)."""

    def __init__(
        self,
        cfg: ModelConfig,
        window_h: int = 24,
        window_w: int = 24,
        mask_mode: str = "span",
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.mask_mode = mask_mode
        self.normalize_targets = normalize_targets
        self.embed = WindowEmbedder(cfg, window_h, window_w)
        self.pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.encoder = Encoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.reconstruct = nn.Linear(cfg.d_model, window_h * window_w)
        self.window_shape = (window_h, window_w)

    def forward(
        self, windows: torch.Tensor, mask_ratio: float = 0.4, generator=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, h, w = windows.shape
        x = self.embed(windows)
        mask = make_mask(self.mask_mode, b, n, mask_ratio, windows.device, generator)
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), x)
        enc = self.encoder(self.pos(x))
        pred = self.reconstruct(enc).reshape(b, n, h, w)
        target = normalize_windows(windows) if self.normalize_targets else windows
        loss = F.mse_loss(pred[mask], target[mask])
        return loss, mask

    def encoder_state(self) -> dict:
        return {"embed": self.embed.state_dict(), "encoder": self.encoder.state_dict()}


class JEPAModel(nn.Module):
    """I-JEPA family: predict target-span REPRESENTATIONS, not pixels.

    Context encoder sees the strip with target spans masked out; a small
    predictor then predicts the target encoder's representations at those
    positions. The target encoder is an EMA copy with stop-gradient — the
    standard anti-collapse construction, backed here by an explicit detector.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        window_h: int = 24,
        window_w: int = 24,
        ema_momentum: float = 0.996,
        predictor_layers: int = 2,
        collapse_ratio: float = 0.1,
        warmup_steps: int = 50,
    ):
        super().__init__()
        self.cfg = cfg
        self.mask_mode = "span"
        self.embed = WindowEmbedder(cfg, window_h, window_w)
        self.pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.encoder = Encoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))

        pred_cfg = ModelConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff=cfg.ff,
            enc_layers=predictor_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            vocab_size=cfg.vocab_size,
        )
        self.predictor = Encoder(pred_cfg, n_layers=predictor_layers)

        self.target_embed = copy.deepcopy(self.embed)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in list(self.target_embed.parameters()) + list(self.target_encoder.parameters()):
            p.requires_grad_(False)

        self.ema_momentum = ema_momentum
        self.collapse_ratio = collapse_ratio
        # must arm inside a 200-step kill-gate, or gates run with no detector
        self.warmup_steps = warmup_steps
        self.register_buffer("_steps", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("_ref_var", torch.zeros((), dtype=torch.float), persistent=False)

    @torch.no_grad()
    def _ema_update(self) -> None:
        m = self.ema_momentum
        for tgt, src in (
            (self.target_embed, self.embed),
            (self.target_encoder, self.encoder),
        ):
            for pt, ps in zip(tgt.parameters(), src.parameters(), strict=True):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            for bt, bs in zip(tgt.buffers(), src.buffers(), strict=True):
                bt.copy_(bs)

    def forward(
        self, windows: torch.Tensor, mask_ratio: float = 0.4, generator=None
    ) -> tuple[torch.Tensor, dict]:
        b, n, _, _ = windows.shape
        if self.training:
            self._ema_update()

        mask = make_mask(self.mask_mode, b, n, mask_ratio, windows.device, generator)

        with torch.no_grad():
            targets = self.target_encoder(self.pos(self.target_embed(windows)))
            targets = F.layer_norm(targets, (targets.size(-1),))

        # collapse detection on the target representations
        var = targets.var(dim=(0, 1)).mean().detach()
        self._steps += 1
        if int(self._steps) == self.warmup_steps:
            self._ref_var = var
        elif (
            int(self._steps) > self.warmup_steps
            and float(self._ref_var) > 0
            and float(var) < self.collapse_ratio * float(self._ref_var)
        ):
            raise RepresentationCollapse(
                f"target representation variance {float(var):.3e} fell below "
                f"{self.collapse_ratio:.0%} of its step-{self.warmup_steps} reference "
                f"{float(self._ref_var):.3e} — encoder has collapsed, aborting"
            )

        ctx = self.embed(windows)
        ctx = torch.where(mask.unsqueeze(-1), self.mask_token.expand(b, n, -1), ctx)
        ctx = self.encoder(self.pos(ctx))
        pred = self.predictor(self.pos(ctx))

        loss = F.smooth_l1_loss(pred[mask], targets[mask])
        # Trivial baseline, same discipline as the MAE arm: what a predictor
        # that always emits the batch-mean target would score. A loss near this
        # value means the arm learned nothing, however pretty the curve.
        with torch.no_grad():
            trivial = F.smooth_l1_loss(
                targets[mask].mean(dim=0, keepdim=True).expand_as(targets[mask]), targets[mask]
            )
        return loss, {
            "target_var": float(var),
            "mask_frac": float(mask.float().mean()),
            "trivial_baseline": float(trivial),
            "loss_over_trivial": float(loss.detach() / (trivial + 1e-9)),
        }

    def encoder_state(self) -> dict:
        return {"embed": self.embed.state_dict(), "encoder": self.encoder.state_dict()}


def warm_start_from_ssl(seq2seq, ssl_state: dict) -> None:
    """Load SSL-pretrained front end + encoder into a pixel Seq2Seq."""
    seq2seq.src_embed.load_state_dict(ssl_state["embed"])
    seq2seq.encoder.load_state_dict(ssl_state["encoder"])
