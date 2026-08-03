"""Sealed known-relative character LMs (§ Phase 3.4).

Decoder-only causal Transformers (~10-25M params at defaults) trained from
scratch, one per known-relative corpus (Hebrew, Greek, Coptic, German
target side). These are the ONLY language models the decipherment track may
consume; training-data manifests live in docs/sealed_models.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from glyphos.models.transformer import (
    Encoder,
    ModelConfig,
    SinusoidalPositions,
    causal_mask,
)

PAD, BOS, EOS, UNK = 0, 1, 2, 3
RESERVED = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharVocab:
    """Character inventory built from the training corpus only (in-repo)."""

    def __init__(self, chars: list[str]):
        self.itos = RESERVED + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, max_chars: int = 4096) -> "CharVocab":
        from collections import Counter

        counts = Counter()
        for t in texts:
            counts.update(t)
        return cls([c for c, _ in counts.most_common(max_chars)])

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in text]
        return [BOS, *ids, EOS] if add_special else ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(RESERVED))

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos[len(RESERVED) :], ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls(json.loads(path.read_text()))


@dataclass(frozen=True)
class CharLMConfig:
    vocab_size: int
    d_model: int = 384
    n_heads: int = 6
    ff: int = 1536
    n_layers: int = 6
    dropout: float = 0.1
    max_len: int = 512


class CharLM(nn.Module):
    def __init__(self, cfg: CharLMConfig):
        super().__init__()
        self.cfg = cfg
        core = ModelConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff=cfg.ff,
            enc_layers=cfg.n_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            vocab_size=cfg.vocab_size,
        )
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.blocks = Encoder(core, n_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.out.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=cfg.d_model**-0.5)
        nn.init.zeros_(self.embed.weight[PAD])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # (B,T) -> (B,T,V)
        x = self.pos(self.embed(ids))
        h = self.blocks(x, attn_mask=causal_mask(ids.size(1), device=ids.device))
        return self.out(h)

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ids[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=PAD
        )

    @torch.no_grad()
    def bits_per_char(self, ids: torch.Tensor) -> float:
        self.eval()
        return float(self.loss(ids) / torch.log(torch.tensor(2.0)))
