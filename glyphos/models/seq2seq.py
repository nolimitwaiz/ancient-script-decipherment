"""Seq2seq models (§ Phase 3.1/3.2): pixel encoder-decoder and the
architecture-matched BPE control, sharing one Transformer core.

Front end differs (WindowEmbedder vs source embedding matrix); everything
downstream — encoder, decoder over target subwords, tied output projection,
label-smoothed loss (0.2 per the recipe) — is identical, so pixel-vs-BPE
comparisons are architecture-matched by construction.
"""

import torch
from torch import nn
from torch.nn import functional as F

from glyphos.models.pixel_encoder import WindowEmbedder
from glyphos.models.transformer import (
    Decoder,
    Encoder,
    ModelConfig,
    SinusoidalPositions,
    causal_mask,
)

PAD, BOS, EOS = 0, 1, 2  # reserved ids in every in-repo vocab


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig, frontend: str):
        super().__init__()
        if frontend not in ("pixel", "tokens"):
            raise ValueError(f"frontend must be pixel|tokens, got {frontend!r}")
        self.cfg = cfg
        self.frontend = frontend
        if frontend == "pixel":
            self.src_embed = WindowEmbedder(cfg)
        else:
            if not cfg.src_vocab_size:
                raise ValueError("tokens frontend requires src_vocab_size")
            self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD)
        self.src_pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.encoder = Encoder(cfg)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.tgt_pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.decoder = Decoder(cfg)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.out.weight = self.tgt_embed.weight  # tied
        for emb in (
            self.tgt_embed,
            *([self.src_embed] if isinstance(self.src_embed, nn.Embedding) else []),
        ):
            nn.init.normal_(emb.weight, std=cfg.d_model**-0.5)
            nn.init.zeros_(emb.weight[PAD])

    def encode(self, src, src_padding_mask=None):
        return self.encoder(self.src_pos(self.src_embed(src)), key_padding_mask=src_padding_mask)

    def forward(self, src, tgt_in, src_padding_mask=None):
        """src: (B,N,H,W) pixel windows or (B,S) token ids; tgt_in: (B,T) with BOS.
        Returns logits (B,T,V)."""
        memory = self.encode(src, src_padding_mask)
        x = self.tgt_pos(self.tgt_embed(tgt_in))
        h = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask(tgt_in.size(1), device=tgt_in.device),
            memory_key_padding_mask=src_padding_mask,
        )
        return self.out(h)

    def loss(self, logits, tgt_out, label_smoothing: float = 0.2):
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=PAD,
            label_smoothing=label_smoothing,
        )

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 64, src_padding_mask=None) -> torch.Tensor:
        self.eval()
        memory = self.encode(src, src_padding_mask)
        batch = memory.size(0)
        ys = torch.full((batch, 1), BOS, dtype=torch.long, device=memory.device)
        done = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        for _ in range(max_len):
            x = self.tgt_pos(self.tgt_embed(ys))
            h = self.decoder(
                x,
                memory,
                tgt_mask=causal_mask(ys.size(1), device=ys.device),
                memory_key_padding_mask=src_padding_mask,
            )
            nxt = self.out(h[:, -1]).argmax(-1, keepdim=True)
            nxt[done] = PAD
            ys = torch.cat([ys, nxt], dim=1)
            done |= nxt.squeeze(1) == EOS
            if bool(done.all()):
                break
        return ys[:, 1:]


def pixel_model(cfg: ModelConfig) -> Seq2Seq:
    return Seq2Seq(cfg, frontend="pixel")


def bpe_model(cfg: ModelConfig) -> Seq2Seq:
    return Seq2Seq(cfg, frontend="tokens")
