"""End-to-end smoke pipeline (§ Phase 0.1: `make smoke`).

Exercises every Phase 0 mechanism in one pass, in the exact shape later phases
must follow:
  data generation -> document-held-out split -> version hashes -> ledger
  preregistration (hypothesis first) -> fit on train -> select on valid ->
  score on locked test (audited by the guard) -> terminal ledger event ->
  self-check that the test read was logged.

The 'science' is deliberately trivial: recover a hidden substitution key by
frequency-rank matching. It runs in well under a second on CPU.
"""

from dataclasses import dataclass, field

from glyphos.align import freqmatch
from glyphos.data import guard, toy
from glyphos.ledger import Ledger
from glyphos.utils import paths
from glyphos.utils.hashing import config_hash
from glyphos.utils.seed import set_seed


class SmokeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeConfig:
    seed: int = 1337
    n_sentences: int = 240
    doc_size: int = 8
    n_word_types: int = 40
    split_ratios: dict = field(default_factory=lambda: {"train": 0.7, "valid": 0.15, "test": 0.15})
    # Regression floor for the deterministic default seed, not a scientific claim.
    min_key_accuracy: float = 0.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def run_smoke(cfg: SmokeConfig, ledger: Ledger | None = None) -> dict:
    ledger = ledger or Ledger()
    guard.install_guard()
    set_seed(cfg.seed)

    print(f"[smoke] data root: {paths.data_root()}")
    print(f"[smoke] ledger:    {ledger.path}")

    corpus = toy.generate_toy_corpus(cfg.n_sentences, cfg.doc_size, cfg.n_word_types, seed=cfg.seed)
    splits = toy.split_by_document(corpus.sentences, cfg.split_ratios, seed=cfg.seed)
    data_version, split_version = toy.write_corpus(corpus, splits, paths.data_root())
    print(
        f"[smoke] corpus written: {len(corpus.sentences)} sentences, "
        f"data_version={data_version}, split_version={split_version}"
    )

    cfg_hash = config_hash(cfg.__dict__)
    with ledger.run(
        hypothesis=(
            "Frequency-rank matching recovers most of the hidden 12-letter substitution "
            f"key from distributional evidence alone (key accuracy >= {cfg.min_key_accuracy})."
        ),
        phase="phase0",
        family="phase0-smoke-toy-decipherment",
        config_hash=cfg_hash,
        data_version=data_version,
        split_version=split_version,
        seed=cfg.seed,
        selection_metric="key_accuracy",
    ) as run:
        print(f"[smoke] registered run {run.run_id} (hypothesis logged before results)")

        split_root = paths.data_root() / "toy" / "splits" / toy.SPLIT_SCHEME / toy.SPLIT_TAG
        train = toy.read_sentences(split_root / "train" / "sentences.jsonl")
        valid = toy.read_sentences(split_root / "valid" / "sentences.jsonl")

        # "Train": estimate the mapping from train-side frequencies only.
        src_freqs = freqmatch.char_frequencies((s.src for s in train), toy.CIPHER_ALPHABET)
        tgt_freqs = freqmatch.char_frequencies((s.tgt for s in train), toy.PLAIN_ALPHABET)
        mapping = freqmatch.rank_match(
            src_freqs, tgt_freqs, toy.CIPHER_ALPHABET, toy.PLAIN_ALPHABET
        )

        valid_acc = freqmatch.token_accuracy(
            [freqmatch.decode(s.src, mapping) for s in valid], [s.tgt for s in valid]
        )
        run.log_metric("valid_token_accuracy", valid_acc)

        # Locked test read — this open() must appear in the guard audit log.
        test = toy.read_sentences(split_root / "test" / "sentences.jsonl")
        test_acc = freqmatch.token_accuracy(
            [freqmatch.decode(s.src, mapping) for s in test], [s.tgt for s in test]
        )
        key_acc = freqmatch.mapping_accuracy(mapping, corpus.key)
        run.log_metrics(
            {
                "key_accuracy": key_acc,
                "test_token_accuracy": test_acc,
                "n_train": len(train),
                "n_valid": len(valid),
                "n_test": len(test),
            }
        )
        print(
            f"[smoke] key_accuracy={key_acc:.3f}  valid_token_acc={valid_acc:.3f}  "
            f"test_token_acc={test_acc:.3f}"
        )

        _require(
            key_acc >= cfg.min_key_accuracy,
            f"key accuracy {key_acc:.3f} below regression floor {cfg.min_key_accuracy}",
        )

        accesses = [
            a
            for a in guard.read_access_log()
            if a["run_id"] == run.run_id and "/test/" in a["path"]
        ]
        _require(
            len(accesses) >= 1,
            "test-set guard did not log the test read — guard is broken",
        )
        print(f"[smoke] guard audited {len(accesses)} test-file read(s) for this run")
        metrics = dict(run.metrics)

    return metrics
