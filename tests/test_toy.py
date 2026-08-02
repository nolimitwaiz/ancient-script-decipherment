import numpy as np
import pytest

from glyphos.align import freqmatch
from glyphos.data import toy


def test_generation_deterministic():
    a = toy.generate_toy_corpus(64, 8, 20, seed=7)
    b = toy.generate_toy_corpus(64, 8, 20, seed=7)
    assert a.sentences == b.sentences
    assert a.key == b.key
    c = toy.generate_toy_corpus(64, 8, 20, seed=8)
    assert c.sentences != a.sentences


def test_cipher_is_consistent_with_key():
    corpus = toy.generate_toy_corpus(32, 8, 20, seed=7)
    plain_to_cipher = {p: c for c, p in corpus.key.items()}
    for s in corpus.sentences:
        assert toy.encipher(s.tgt, plain_to_cipher) == s.src


def test_document_split_disjoint_and_complete():
    corpus = toy.generate_toy_corpus(240, 8, 40, seed=7)
    splits = toy.split_by_document(
        corpus.sentences, {"train": 0.7, "valid": 0.15, "test": 0.15}, seed=7
    )
    docs = {name: {s.doc_id for s in part} for name, part in splits.items()}
    assert not (docs["train"] & docs["valid"])
    assert not (docs["train"] & docs["test"])
    assert not (docs["valid"] & docs["test"])
    assert sum(len(p) for p in splits.values()) == len(corpus.sentences)


def test_split_rejects_bad_ratios():
    corpus = toy.generate_toy_corpus(64, 8, 20, seed=7)
    with pytest.raises(ValueError, match="sum to 1"):
        toy.split_by_document(corpus.sentences, {"train": 0.9, "valid": 0.2, "test": 0.1}, seed=7)
    with pytest.raises(ValueError, match="train/valid/test"):
        toy.split_by_document(corpus.sentences, {"train": 1.0}, seed=7)


def test_write_corpus_versions_are_content_hashes(tmp_path):
    corpus = toy.generate_toy_corpus(64, 8, 20, seed=7)
    splits = toy.split_by_document(
        corpus.sentences, {"train": 0.7, "valid": 0.15, "test": 0.15}, seed=7
    )
    dv1, sv1 = toy.write_corpus(corpus, splits, tmp_path / "d1")
    dv2, sv2 = toy.write_corpus(corpus, splits, tmp_path / "d2")
    assert (dv1, sv1) == (dv2, sv2)
    test_dir = tmp_path / "d1/toy/splits" / toy.SPLIT_SCHEME / toy.SPLIT_TAG / "test"
    assert (test_dir / "sentences.jsonl").exists()
    assert toy.read_sentences(test_dir / "sentences.jsonl") == splits["test"]


# -- frequency matching -----------------------------------------------------


def test_char_frequencies_normalized():
    freqs = freqmatch.char_frequencies(["aab", "b c"], ["a", "b", "c"])
    assert freqs == pytest.approx([2 / 5, 2 / 5, 1 / 5])
    with pytest.raises(ValueError, match="no in-alphabet"):
        freqmatch.char_frequencies(["xyz"], ["a"])


def test_rank_match_and_decode():
    mapping = freqmatch.rank_match(
        np.array([0.1, 0.7, 0.2]), np.array([0.7, 0.1, 0.2]), ["A", "B", "C"], ["a", "b", "c"]
    )
    assert mapping == {"B": "a", "C": "c", "A": "b"}
    assert freqmatch.decode("BCA B", mapping) == "acb a"


def test_mapping_accuracy_requires_same_inventory():
    with pytest.raises(ValueError, match="inventories"):
        freqmatch.mapping_accuracy({"A": "a"}, {"A": "a", "B": "b"})
    assert freqmatch.mapping_accuracy({"A": "a", "B": "x"}, {"A": "a", "B": "b"}) == 0.5


def test_frequency_matching_beats_chance_on_toy_corpus():
    corpus = toy.generate_toy_corpus(240, 8, 40, seed=1337)
    src = freqmatch.char_frequencies((s.src for s in corpus.sentences), toy.CIPHER_ALPHABET)
    tgt = freqmatch.char_frequencies((s.tgt for s in corpus.sentences), toy.PLAIN_ALPHABET)
    mapping = freqmatch.rank_match(src, tgt, toy.CIPHER_ALPHABET, toy.PLAIN_ALPHABET)
    # 12 symbols: chance ~ 1/12 correct; parallel-text frequencies must do far better.
    assert freqmatch.mapping_accuracy(mapping, corpus.key) >= 0.5
