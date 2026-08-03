"""From-scratch Transformer blocks (§ Phase 3) — every parameter randomly
initialized here; no pretrained weights anywhere (hard constraint).

Pre-LayerNorm blocks (stable without warmup gymnastics), sinusoidal
positions, explicit encoder/decoder stacks assembled by hand so the
architecture stays inspectable. Defaults follow Salesky et al.'s best TED
config: deep encoder / shallow decoder (12/3), d=512, FF 4096, 4 heads.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    n_heads: int = 4
    ff: int = 4096
    enc_layers: int = 12
    dec_layers: int = 3
    dropout: float = 0.1
    max_len: int = 1024
    vocab_size: int = 10_000  # target subwords
    src_vocab_size: int | None = None  # BPE baseline only; None for pixel front-end
    conv_channels: int = 32  # pixel front-end only


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return x + self.pe[: x.size(1)]


class _FeedForward(nn.Sequential):
    def __init__(self, cfg: ModelConfig):
        super().__init__(
            nn.Linear(cfg.d_model, cfg.ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff, cfg.d_model),
        )


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = _FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.ff = _FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=tgt_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        a, _ = self.cross_attn(
            h, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig, n_layers: int | None = None):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(n_layers or cfg.enc_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(x)


def causal_mask(size: int, device=None) -> torch.Tensor:
    """Float mask with -inf above the diagonal (additive attention mask)."""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
