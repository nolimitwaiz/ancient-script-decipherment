"""Model smoke (§ Phase 3): every model family trains 50 steps on toy data on
CPU inside `make smoke`, and its loss must fall. One ledger run records all
four losses.

Pixel windows come from a deterministic synthetic bitmap font (hash-derived
24x24 glyphs) so the smoke needs no cairo backend; the real renderer is
exercised separately in the render environment.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from glyphos.data import toy
from glyphos.ledger import Ledger
from glyphos.models.char_lm import CharLM, CharLMConfig, CharVocab
from glyphos.models.seq2seq import EOS, PAD, Seq2Seq
from glyphos.models.ssl import MaskedWindowModel
from glyphos.models.tokenizer import Subwords, train_sentencepiece
from glyphos.models.transformer import ModelConfig, count_params
from glyphos.train import TrainConfig, fit
from glyphos.utils import paths
from glyphos.utils.hashing import config_hash
from glyphos.utils.seed import set_seed


@dataclass(frozen=True)
class ModelSmokeConfig:
    seed: int = 1337
    steps: int = 50
    batch: int = 8
    d_model: int = 64
    n_heads: int = 2
    ff: int = 128
    enc_layers: int = 2
    dec_layers: int = 1
    sp_vocab: int = 200
    min_improvement: float = 0.0  # end loss must be strictly below start loss


class ModelSmokeFailure(RuntimeError):
    pass


def _tiny_cfg(cfg: ModelSmokeConfig, vocab: int, src_vocab: int | None = None) -> ModelConfig:
    return ModelConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        ff=cfg.ff,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        dropout=0.1,
        max_len=256,
        vocab_size=vocab,
        src_vocab_size=src_vocab,
        conv_channels=8,
    )


def _bitmap_glyph(ch: str) -> np.ndarray:
    """Deterministic fake 24x24 glyph from the character's hash."""
    seed = int.from_bytes(hashlib.sha256(ch.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    glyph = np.ones((24, 24), dtype=np.float32)
    for _ in range(6):  # a few random strokes
        y, x = rng.integers(2, 22, 2)
        dy, dx = rng.integers(-2, 3, 2)
        for t in range(10):
            yy, xx = int(np.clip(y + dy * t / 3, 0, 23)), int(np.clip(x + dx * t / 3, 0, 23))
            glyph[max(0, yy - 1) : yy + 2, max(0, xx - 1) : xx + 2] = 0.0
    return glyph


def _windows_for(text: str, max_windows: int = 24) -> np.ndarray:
    glyphs = [_bitmap_glyph(c) for c in text.replace(" ", "")[:max_windows]]
    if not glyphs:
        glyphs = [_bitmap_glyph(" ")]
    return np.stack(glyphs)


def _pad_batch_ids(seqs: list[list[int]], length: int) -> torch.Tensor:
    out = torch.full((len(seqs), length), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:length]
        out[i, : len(s)] = torch.tensor(s)
    return out


def _seq2seq_stream(pairs, encode_tgt, cfg: ModelSmokeConfig, pixel: bool, src_encode=None):
    rng = np.random.default_rng(cfg.seed)
    while True:
        idx = rng.integers(0, len(pairs), cfg.batch)
        src_txt = [pairs[i][0] for i in idx]
        tgt_ids = [encode_tgt(pairs[i][1]) for i in idx]
        tgt = _pad_batch_ids(tgt_ids, 48)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        if pixel:
            wins = [_windows_for(s) for s in src_txt]
            n = max(w.shape[0] for w in wins)
            batch_w = np.ones((cfg.batch, n, 24, 24), dtype=np.float32)
            for i, w in enumerate(wins):
                batch_w[i, : w.shape[0]] = w
            yield torch.from_numpy(batch_w), tgt_in, tgt_out
        else:
            src = _pad_batch_ids([src_encode(s) for s in src_txt], 48)
            yield src, tgt_in, tgt_out


def _fit_50(model, batches, loss_fn, cfg: ModelSmokeConfig, name: str) -> tuple[float, float]:
    result = fit(
        model,
        batches,
        loss_fn,
        TrainConfig(
            lr=3e-4,
            warmup_steps=10,
            max_steps=cfg.steps,
            eval_every=10**9,  # no eval inside smoke
            log_every=10**9,
            device="cpu",
            checkpoint_dir=None,
            seed=cfg.seed,
        ),
    )
    if not result.last_loss < result.first_loss - cfg.min_improvement:
        raise ModelSmokeFailure(
            f"{name}: loss did not fall in {cfg.steps} steps "
            f"({result.first_loss:.3f} -> {result.last_loss:.3f})"
        )
    return result.first_loss, result.last_loss


def run_model_smoke(cfg: ModelSmokeConfig, ledger: Ledger | None = None) -> dict:
    ledger = ledger or Ledger()
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    corpus = toy.generate_toy_corpus(120, 8, 30, seed=cfg.seed)
    pairs = [(s.src, s.tgt) for s in corpus.sentences]
    plain_texts = [t for _, t in pairs]

    sp_path = paths.runs_dir() / "smoke" / "sp" / "toy_tgt"
    sp_model = train_sentencepiece(plain_texts, sp_path, vocab_size=cfg.sp_vocab)
    subwords = Subwords(sp_model)

    metrics: dict = {"n_pairs": len(pairs), "sp_vocab": len(subwords)}
    with ledger.run(
        hypothesis=(
            "All four from-scratch model families (char-LM, BPE seq2seq, pixel "
            "seq2seq, masked-window SSL) reduce training loss within 50 CPU steps "
            "on toy data."
        ),
        phase="phase3",
        family="phase3-smoke-models",
        config_hash=config_hash(cfg.__dict__),
        data_version=f"toy-{cfg.seed}",
        split_version="smoke",
        seed=cfg.seed,
        selection_metric="pixel_last_loss",
    ) as run:
        # 1. sealed char LM on toy plaintext
        vocab = CharVocab.build(plain_texts)
        lm = CharLM(
            CharLMConfig(
                vocab_size=len(vocab),
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff=cfg.ff,
                n_layers=cfg.enc_layers,
                max_len=128,
            )
        )

        def lm_batches():
            rng = np.random.default_rng(cfg.seed)
            while True:
                idx = rng.integers(0, len(plain_texts), cfg.batch)
                yield _pad_batch_ids([vocab.encode(plain_texts[i])[:64] for i in idx], 64)

        first, last = _fit_50(lm, lm_batches(), lambda m, b: m.loss(b), cfg, "char_lm")
        metrics.update(
            {
                "char_lm_first_loss": first,
                "char_lm_last_loss": last,
                "char_lm_params": count_params(lm),
            }
        )
        print(f"[smoke] char_lm: {first:.3f} -> {last:.3f} ({count_params(lm):,} params)")

        # 2. BPE control seq2seq
        src_vocab = CharVocab.build([s for s, _ in pairs])
        bpe = Seq2Seq(_tiny_cfg(cfg, len(subwords), src_vocab=len(src_vocab)), frontend="tokens")
        stream = _seq2seq_stream(
            pairs, subwords.encode, cfg, pixel=False, src_encode=src_vocab.encode
        )
        first, last = _fit_50(
            bpe, stream, lambda m, b: m.loss(m(b[0], b[1]), b[2]), cfg, "bpe_seq2seq"
        )
        metrics.update(
            {"bpe_first_loss": first, "bpe_last_loss": last, "bpe_params": count_params(bpe)}
        )
        print(f"[smoke] bpe_seq2seq: {first:.3f} -> {last:.3f} ({count_params(bpe):,} params)")

        # 3. pixel seq2seq on synthetic bitmap windows
        pix = Seq2Seq(_tiny_cfg(cfg, len(subwords)), frontend="pixel")
        stream = _seq2seq_stream(pairs, subwords.encode, cfg, pixel=True)
        first, last = _fit_50(
            pix, stream, lambda m, b: m.loss(m(b[0], b[1]), b[2]), cfg, "pixel_seq2seq"
        )
        metrics.update(
            {"pixel_first_loss": first, "pixel_last_loss": last, "pixel_params": count_params(pix)}
        )
        print(f"[smoke] pixel_seq2seq: {first:.3f} -> {last:.3f} ({count_params(pix):,} params)")

        # 4. masked-window SSL
        ssl = MaskedWindowModel(_tiny_cfg(cfg, len(subwords)))

        def ssl_batches():
            stream_inner = _seq2seq_stream(pairs, subwords.encode, cfg, pixel=True)
            while True:
                yield next(stream_inner)[0]

        first, last = _fit_50(ssl, ssl_batches(), lambda m, b: m(b)[0], cfg, "ssl")
        metrics.update(
            {"ssl_first_loss": first, "ssl_last_loss": last, "ssl_params": count_params(ssl)}
        )
        print(f"[smoke] ssl: {first:.3f} -> {last:.3f} ({count_params(ssl):,} params)")

        # greedy decode round-trips shapes
        sample = next(_seq2seq_stream(pairs, subwords.encode, cfg, pixel=True))[0][:2]
        decoded = pix.greedy_decode(sample, max_len=8)
        if decoded.shape[0] != 2 or decoded.shape[1] < 1:
            raise ModelSmokeFailure(f"greedy_decode bad shape {tuple(decoded.shape)}")
        metrics["decode_len"] = int(decoded.shape[1])

        run.log_metrics(metrics)
    return metrics


_ = EOS  # re-exported convention check: reserved ids stay 0/1/2 repo-wide
