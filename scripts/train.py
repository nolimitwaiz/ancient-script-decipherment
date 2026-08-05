#!/usr/bin/env python
"""Training entry point (§ Phase 3): config-driven, ledger-preregistered.

    train.py --config configs/train/sealed_lm_hebrew.yaml [--kill-gate]

Tasks: char_lm | translation_bpe | translation_pixel | ssl.
Data comes ONLY from frozen-split train/valid partitions (the guard audits
every read; test partitions are never touched here). The run registers in
the ledger — hypothesis required, straight from the config — before the
first step. --kill-gate caps the run at `kill_gate_steps` for the mandatory
one-budget-unit sanity run before any long job.
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass

import numpy as np
import torch

from glyphos.data.guard import install_guard
from glyphos.data.ingest import processed_dir
from glyphos.data.schema import Record, read_records
from glyphos.ledger import Ledger
from glyphos.models.char_lm import CharLM, CharLMConfig, CharVocab
from glyphos.models.seq2seq import PAD, Seq2Seq
from glyphos.models.ssl import MaskedWindowModel
from glyphos.models.tokenizer import Subwords, train_sentencepiece
from glyphos.models.transformer import ModelConfig, count_params
from glyphos.train import TrainConfig, fit
from glyphos.utils import paths
from glyphos.utils.config import load_config
from glyphos.utils.hashing import config_hash
from glyphos.utils.seed import set_seed

TASKS = ("char_lm", "translation_bpe", "translation_pixel", "ssl")


@dataclass(frozen=True)
class TrainRunConfig:
    task: str
    corpora: list  # one or more corpus names; concatenated for LM/SSL
    hypothesis: str
    family: str
    scheme: str = "dedup"
    tag: str = "v1"
    seed: int = 1337
    src_field: str | None = None  # defaults from corpus manifest
    tgt_field: str | None = None
    # model
    d_model: int = 512
    n_heads: int = 4
    ff: int = 4096
    enc_layers: int = 12
    dec_layers: int = 3
    dropout: float = 0.1
    lm_d_model: int = 384
    lm_layers: int = 6
    lm_n_heads: int = 6
    sp_vocab_tgt: int = 10_000
    sp_vocab_src: int = 4_000
    # training
    lr: float = 5e-4
    warmup_steps: int = 4000
    max_steps: int = 100_000
    batch_sentences: int = 64
    grad_accum: int = 4
    eval_every: int = 1000
    patience: int = 10
    max_src_len: int = 256
    max_tgt_len: int = 128
    kill_gate_steps: int = 200
    device: str = "auto"
    warm_start_ssl: str | None = None  # path to an in-repo SSL checkpoint

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if not str(self.hypothesis).strip():
            raise ValueError("hypothesis is required (preregistration, CONVENTIONS.md §2)")


def _split_dir(corpus: str, cfg: TrainRunConfig):
    return paths.data_root() / corpus / "splits" / cfg.scheme / cfg.tag


def _load_partition(cfg: TrainRunConfig, part: str) -> list[Record]:
    records: list[Record] = []
    for corpus in cfg.corpora:
        records.extend(read_records(_split_dir(corpus, cfg) / part / "records.jsonl"))
    if not records:
        raise SystemExit(f"no records in {part} for {cfg.corpora} ({cfg.scheme}/{cfg.tag})")
    return records


def _manifest(corpus: str) -> dict:
    with open(processed_dir(corpus) / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def _versions(cfg: TrainRunConfig) -> tuple[str, str]:
    data_v = "+".join(_manifest(c)["data_version"] for c in cfg.corpora)
    split_v = []
    for corpus in cfg.corpora:
        with open(_split_dir(corpus, cfg) / "split_info.json", encoding="utf-8") as f:
            split_v.append(json.load(f)["split_version"])
    return data_v, "+".join(split_v)


def _pad(seqs: list[list[int]], length: int) -> torch.Tensor:
    out = torch.full((len(seqs), length), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : min(len(s), length)] = torch.tensor(s[:length])
    return out


def _texts(records: list[Record], fld: str) -> list[str]:
    return [r.fields[fld] for r in records if fld in r.fields and r.fields[fld].strip()]


# -- task builders ----------------------------------------------------------


def build_char_lm(cfg, train_recs, valid_recs, run_dir):
    fld = cfg.src_field or _manifest(cfg.corpora[0])["primary_field"]
    train_texts, valid_texts = _texts(train_recs, fld), _texts(valid_recs, fld)
    vocab = CharVocab.build(train_texts)
    vocab.save(run_dir / "char_vocab.json")
    model = CharLM(
        CharLMConfig(
            vocab_size=len(vocab),
            d_model=cfg.lm_d_model,
            n_heads=cfg.lm_n_heads,
            n_layers=cfg.lm_layers,
            max_len=cfg.max_src_len,
        )
    )

    def batches():
        rng = np.random.default_rng(cfg.seed)
        while True:
            idx = rng.integers(0, len(train_texts), cfg.batch_sentences)
            yield _pad([vocab.encode(train_texts[i]) for i in idx], cfg.max_src_len)

    eval_ids = _pad([vocab.encode(t) for t in valid_texts[:512]], cfg.max_src_len)

    def eval_fn(m):
        with torch.no_grad():
            return float(m.loss(eval_ids.to(next(m.parameters()).device)))

    return model, batches(), (lambda m, b: m.loss(b)), eval_fn


def _window_batcher(texts, cfg, rcfg, font):
    from glyphos.render.renderer import render_strip_cached
    from glyphos.render.windows import slice_windows

    def encode(batch_texts):
        wins = [slice_windows(render_strip_cached(t, rcfg, font), rcfg) for t in batch_texts]
        n = min(max(w.shape[0] for w in wins), cfg.max_src_len)
        batch = np.ones((len(wins), n, rcfg.window_h, rcfg.window_w), dtype=np.float32)
        for i, w in enumerate(wins):
            batch[i, : min(w.shape[0], n)] = w[:n]
        return torch.from_numpy(batch)

    return encode


def build_translation(cfg, train_recs, valid_recs, run_dir, pixel: bool):
    manifest = _manifest(cfg.corpora[0])
    src_fld = cfg.src_field or manifest["primary_field"]
    tgt_fld = cfg.tgt_field or manifest["translation_field"]

    def _pairs(records):
        return [
            (r.fields[src_fld], r.fields[tgt_fld])
            for r in records
            if src_fld in r.fields and tgt_fld in r.fields
        ]

    pairs, vpairs = _pairs(train_recs), _pairs(valid_recs)
    sp_tgt = Subwords(
        train_sentencepiece((t for _, t in pairs), run_dir / "sp_tgt", cfg.sp_vocab_tgt)
    )

    if pixel:
        from glyphos.render.config import RenderConfig
        from glyphos.render.renderer import SCRIPT_FONTS

        rcfg = RenderConfig()
        script = "egyptian" if "hiero" in src_fld else "default"
        encode_src = _window_batcher(None, cfg, rcfg, SCRIPT_FONTS[script])
        model = Seq2Seq(_model_cfg(cfg, len(sp_tgt), None), frontend="pixel")
        if cfg.warm_start_ssl:
            from glyphos.models.ssl import warm_start_from_ssl

            state = torch.load(cfg.warm_start_ssl, map_location="cpu", weights_only=True)
            warm_start_from_ssl(model, state["ssl_encoder"])
            print(f"[train] warm-started encoder from {cfg.warm_start_ssl}")
    else:
        sp_src = Subwords(
            train_sentencepiece((s for s, _ in pairs), run_dir / "sp_src", cfg.sp_vocab_src)
        )

        def encode_src(batch_texts):
            return _pad([sp_src.encode(t) for t in batch_texts], cfg.max_src_len)

        model = Seq2Seq(_model_cfg(cfg, len(sp_tgt), len(sp_src)), frontend="tokens")

    def batches():
        rng = np.random.default_rng(cfg.seed)
        while True:
            idx = rng.integers(0, len(pairs), cfg.batch_sentences)
            src = encode_src([pairs[i][0] for i in idx])
            tgt = _pad([sp_tgt.encode(pairs[i][1]) for i in idx], cfg.max_tgt_len)
            yield src, tgt[:, :-1], tgt[:, 1:]

    ev = vpairs[:256]

    def eval_fn(m):
        device = next(m.parameters()).device
        with torch.no_grad():
            src = encode_src([s for s, _ in ev]).to(device)
            tgt = _pad([sp_tgt.encode(t) for _, t in ev], cfg.max_tgt_len).to(device)
            return float(m.loss(m(src, tgt[:, :-1]), tgt[:, 1:], label_smoothing=0.0))

    return model, batches(), (lambda m, b: m.loss(m(b[0], b[1]), b[2])), eval_fn


def build_ssl(cfg, train_recs, valid_recs, run_dir):
    from glyphos.render.config import RenderConfig
    from glyphos.render.renderer import SCRIPT_FONTS

    rcfg = RenderConfig()

    # per-record font: hieroglyphic corpora render with the Egyptian font,
    # everything else with the default (Pango falls back per script anyway)
    def _font_for(corpus: str) -> str:
        if corpus.startswith("tla"):
            return SCRIPT_FONTS["egyptian"]
        for key in ("coptic", "linear_b", "cuneiform", "meroitic"):
            if key in corpus:
                return SCRIPT_FONTS.get(key, SCRIPT_FONTS["default"])
        return SCRIPT_FONTS["default"]

    items = []
    for rec in train_recs:
        fld = cfg.src_field or _manifest(rec.corpus)["primary_field"]
        if fld in rec.fields and rec.fields[fld].strip():
            items.append((rec.fields[fld], _font_for(rec.corpus)))
    model = MaskedWindowModel(_model_cfg(cfg, 8, None))

    def batches():
        rng = np.random.default_rng(cfg.seed)
        encoders = {}
        while True:
            idx = rng.integers(0, len(items), cfg.batch_sentences)
            by_font: dict = {}
            for i in idx:
                text, font = items[int(i)]
                by_font.setdefault(font, []).append(text)
            parts = []
            for font, texts_f in by_font.items():
                enc = encoders.setdefault(font, _window_batcher(None, cfg, rcfg, font))
                parts.append(enc(texts_f))
            n = max(p.shape[1] for p in parts)
            import torch as _t

            padded = [
                _t.nn.functional.pad(p, (0, 0, 0, 0, 0, n - p.shape[1]), value=1.0) for p in parts
            ]
            yield _t.cat(padded, dim=0)

    return model, batches(), (lambda m, b: m(b)[0]), None


def _model_cfg(cfg: TrainRunConfig, vocab: int, src_vocab: int | None) -> ModelConfig:
    return ModelConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        ff=cfg.ff,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        dropout=cfg.dropout,
        max_len=max(cfg.max_src_len, cfg.max_tgt_len) + 8,
        vocab_size=vocab,
        src_vocab_size=src_vocab,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--kill-gate", action="store_true", help="cap at kill_gate_steps")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    install_guard()
    cfg = load_config(TrainRunConfig, args.config)
    set_seed(cfg.seed)
    max_steps = cfg.kill_gate_steps if args.kill_gate else cfg.max_steps

    train_recs = _load_partition(cfg, "train")
    valid_recs = _load_partition(cfg, "valid")
    data_v, split_v = _versions(cfg)
    print(
        f"[train] {cfg.task} on {cfg.corpora}: "
        f"{len(train_recs):,} train / {len(valid_recs):,} valid"
    )

    ledger = Ledger()
    with ledger.run(
        hypothesis=cfg.hypothesis + (" [KILL-GATE]" if args.kill_gate else ""),
        phase="phase3",
        family=cfg.family,
        config_hash=config_hash(dataclasses.asdict(cfg)),
        data_version=data_v,
        split_version=split_v,
        seed=cfg.seed,
        selection_metric="best_eval",
    ) as run:
        run_dir = paths.runs_dir() / "checkpoints" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

        builders = {
            "char_lm": build_char_lm,
            "translation_bpe": lambda *a: build_translation(*a, pixel=False),
            "translation_pixel": lambda *a: build_translation(*a, pixel=True),
            "ssl": build_ssl,
        }
        model, batches, loss_fn, eval_fn = builders[cfg.task](cfg, train_recs, valid_recs, run_dir)
        print(f"[train] {count_params(model):,} params -> {run_dir}")

        result = fit(
            model,
            batches,
            loss_fn,
            TrainConfig(
                lr=cfg.lr,
                warmup_steps=min(cfg.warmup_steps, max(10, max_steps // 10)),
                max_steps=max_steps,
                grad_accum=cfg.grad_accum,
                eval_every=min(cfg.eval_every, max(10, max_steps // 4)),
                patience=cfg.patience,
                seed=cfg.seed,
                device=args.device or cfg.device,
                checkpoint_dir=str(run_dir),
            ),
            eval_fn=eval_fn,
            on_metric=lambda k, v: print(f"[train] {k}={v:.4f}"),
        )
        if cfg.task == "ssl":
            torch.save({"ssl_encoder": model.encoder_state()}, run_dir / "ssl_encoder.pt")
        run.log_metrics(
            {
                "params": count_params(model),
                "steps": result.steps,
                "first_loss": result.first_loss,
                "last_loss": result.last_loss,
                "best_eval": result.best_eval,
                "stopped_early": result.stopped_early,
                "kill_gate": args.kill_gate,
            }
        )
        print(
            f"[train] done: {result.steps} steps, loss {result.first_loss:.3f} -> "
            f"{result.last_loss:.3f}, best_eval={result.best_eval:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
