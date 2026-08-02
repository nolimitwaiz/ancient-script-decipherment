"""Deterministic toy substitution-cipher corpus for the smoke pipeline.

Smallest instance of the artificial-decipherment protocol (§ Phase 5.2): a
known 'plaintext language' with Zipfian wordforms is enciphered by a hidden
letter permutation. The pipeline must recover the key from distributional
evidence alone; the key file exists only for final scoring.

Layout written by `write_corpus` follows the repo data contract, so the test
partition lands under a `test/` directory and trips the access guard:
    <data_root>/toy/splits/document_heldout/v1/{train,valid,test}/sentences.jsonl
    <data_root>/toy/private/key.json
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glyphos.utils.hashing import config_hash, hash_dir

PLAIN_ALPHABET = tuple("abcdefghijkl")
CIPHER_ALPHABET = tuple("ABCDEFGHIJKL")
SPLIT_SCHEME = "document_heldout"
SPLIT_TAG = "v1"


@dataclass(frozen=True)
class ToySentence:
    doc_id: str
    src: str  # ciphertext ("lost script")
    tgt: str  # plaintext ("known relative")


@dataclass
class ToyCorpus:
    sentences: list[ToySentence]
    key: dict[str, str]  # cipher -> plain; HIDDEN from everything except final scoring


def _zipf_weights(n: int, exponent: float = 1.1) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** exponent
    return w / w.sum()


def _word_types(rng: np.random.Generator, n_types: int, letter_probs: np.ndarray) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    while len(words) < n_types:
        length = int(rng.integers(2, 7))
        word = "".join(rng.choice(PLAIN_ALPHABET, size=length, p=letter_probs))
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words


def generate_toy_corpus(
    n_sentences: int, doc_size: int, n_word_types: int, seed: int
) -> ToyCorpus:
    rng = np.random.default_rng(seed)
    # Skewed letter distribution -> well-separated unigram frequencies, so the
    # frequency-rank baseline has a real (but imperfect) signal to exploit.
    letter_probs = rng.dirichlet(np.linspace(4.0, 0.4, len(PLAIN_ALPHABET)))
    words = _word_types(rng, n_word_types, letter_probs)
    zipf = _zipf_weights(n_word_types)

    perm = rng.permutation(len(PLAIN_ALPHABET))
    plain_to_cipher = {PLAIN_ALPHABET[i]: CIPHER_ALPHABET[perm[i]] for i in range(len(perm))}
    key = {cipher: plain for plain, cipher in plain_to_cipher.items()}

    sentences = []
    for i in range(n_sentences):
        n_words = int(rng.integers(4, 10))
        tgt = " ".join(rng.choice(words, size=n_words, p=zipf))
        src = encipher(tgt, plain_to_cipher)
        sentences.append(ToySentence(doc_id=f"doc{i // doc_size:04d}", src=src, tgt=tgt))
    return ToyCorpus(sentences=sentences, key=key)


def encipher(text: str, plain_to_cipher: dict[str, str]) -> str:
    return "".join(plain_to_cipher.get(ch, ch) for ch in text)


def split_by_document(
    sentences: list[ToySentence], ratios: dict[str, float], seed: int
) -> dict[str, list[ToySentence]]:
    """Document-held-out split: no doc_id crosses a partition boundary."""
    if set(ratios) != {"train", "valid", "test"}:
        raise ValueError(f"ratios must define train/valid/test, got {sorted(ratios)}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    doc_ids = sorted({s.doc_id for s in sentences})
    rng = np.random.default_rng(seed)
    rng.shuffle(doc_ids)
    n = len(doc_ids)
    n_train = round(n * ratios["train"])
    n_valid = round(n * ratios["valid"])
    assignment = {
        "train": set(doc_ids[:n_train]),
        "valid": set(doc_ids[n_train : n_train + n_valid]),
        "test": set(doc_ids[n_train + n_valid :]),
    }
    splits = {
        name: [s for s in sentences if s.doc_id in docs] for name, docs in assignment.items()
    }
    empty = [name for name, part in splits.items() if not part]
    if empty:
        raise ValueError(f"empty split partition(s) {empty}: corpus too small for {ratios}")
    return splits


def write_corpus(
    corpus: ToyCorpus, splits: dict[str, list[ToySentence]], data_root: Path
) -> tuple[str, str]:
    """Write splits + hidden key; return (data_version, split_version) hashes."""
    corpus_root = data_root / "toy"
    for name, part in splits.items():
        part_dir = corpus_root / "splits" / SPLIT_SCHEME / SPLIT_TAG / name
        part_dir.mkdir(parents=True, exist_ok=True)
        with open(part_dir / "sentences.jsonl", "w", encoding="utf-8") as f:
            for s in part:
                f.write(json.dumps(s.__dict__, sort_keys=True, ensure_ascii=False) + "\n")
    private_dir = corpus_root / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    with open(private_dir / "key.json", "w", encoding="utf-8") as f:
        json.dump(corpus.key, f, sort_keys=True, indent=2)

    data_version = hash_dir(corpus_root)
    split_version = config_hash(
        {name: sorted({s.doc_id for s in part}) for name, part in splits.items()}
    )
    return data_version, split_version


def read_sentences(path: Path) -> list[ToySentence]:
    """Read a sentences.jsonl through plain open(), so guard auditing applies."""
    with open(path, encoding="utf-8") as f:
        return [ToySentence(**json.loads(line)) for line in f if line.strip()]
