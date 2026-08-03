"""In-repo SentencePiece training (§ Phase 3.1) — a tokenizer TRAINER run on
our own corpus text, never a downloaded tokenizer. Unigram model, target-side
text only for translation targets (e.g. German), ids offset so 0/1/2 remain
pad/bos/eos repo-wide.
"""

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def train_sentencepiece(
    texts: Iterable[str],
    model_prefix: Path,
    vocab_size: int = 10_000,
    character_coverage: float = 1.0,
) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_file = model_prefix.with_suffix(".train.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        unk_id=UNK,
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


class Subwords:
    def __init__(self, model_path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        return [BOS, *ids, EOS] if add_special else ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i not in (PAD, BOS, EOS)])

    def __len__(self) -> int:
        return self.sp.get_piece_size()
