"""Model-stack tests: tiny dims, CPU, fast. The 50-step learning check lives
in the smoke (tasks/model_smoke.py); here we test structure and invariants."""

import numpy as np
import pytest
import torch

from glyphos.models.char_lm import CharLM, CharLMConfig, CharVocab
from glyphos.models.seq2seq import PAD, Seq2Seq
from glyphos.models.ssl import MaskedWindowModel, warm_start_from_ssl
from glyphos.models.tokenizer import Subwords, train_sentencepiece
from glyphos.models.transformer import ModelConfig, count_params
from glyphos.train.loop import TrainConfig, load_checkpoint, lr_at, pick_device

TINY = ModelConfig(
    d_model=32,
    n_heads=2,
    ff=64,
    enc_layers=2,
    dec_layers=1,
    max_len=64,
    vocab_size=50,
    src_vocab_size=40,
    conv_channels=4,
)


def test_pixel_and_bpe_shapes():
    torch.manual_seed(0)
    pix = Seq2Seq(TINY, frontend="pixel")
    windows = torch.rand(3, 5, 24, 24)
    tgt_in = torch.randint(3, 50, (3, 7))
    assert pix(windows, tgt_in).shape == (3, 7, 50)

    bpe = Seq2Seq(TINY, frontend="tokens")
    src = torch.randint(3, 40, (3, 9))
    assert bpe(src, tgt_in).shape == (3, 7, 50)


def test_decoder_is_causal():
    """Changing a future target token must not change earlier logits."""
    torch.manual_seed(0)
    model = Seq2Seq(TINY, frontend="tokens").eval()
    src = torch.randint(3, 40, (1, 6))
    tgt = torch.randint(3, 50, (1, 8))
    with torch.no_grad():
        base = model(src, tgt)
        mutated = tgt.clone()
        mutated[0, -1] = (mutated[0, -1] + 1) % 50
        out = model(src, mutated)
    torch.testing.assert_close(base[:, :-1], out[:, :-1])
    assert not torch.allclose(base[:, -1], out[:, -1])


def test_char_lm_is_causal_and_loss_reasonable():
    torch.manual_seed(0)
    vocab = CharVocab.build(["abcabc", "bca"])
    lm = CharLM(CharLMConfig(vocab_size=len(vocab), d_model=32, n_heads=2, ff=64, n_layers=2))
    ids = torch.tensor([vocab.encode("abcabc")])
    loss = lm.loss(ids)
    assert 0 < float(loss) < 2 * np.log(len(vocab))


def test_greedy_decode_shapes_and_determinism():
    torch.manual_seed(0)
    model = Seq2Seq(TINY, frontend="tokens").eval()
    src = torch.randint(3, 40, (2, 6))
    a = model.greedy_decode(src, max_len=5)
    b = model.greedy_decode(src, max_len=5)
    assert a.shape[0] == 2 and 1 <= a.shape[1] <= 5
    torch.testing.assert_close(a, b)


def test_padding_ignored_in_loss():
    torch.manual_seed(0)
    model = Seq2Seq(TINY, frontend="tokens")
    logits = torch.randn(2, 4, 50)
    tgt = torch.full((2, 4), PAD)
    tgt[0, 0] = 5
    loss_padded = model.loss(logits, tgt)
    loss_single = model.loss(logits[:1, :1], tgt[:1, :1])
    torch.testing.assert_close(loss_padded, loss_single)


def test_ssl_masking_and_warm_start():
    torch.manual_seed(0)
    ssl = MaskedWindowModel(TINY)
    windows = torch.rand(2, 6, 24, 24)
    loss, mask = ssl(windows, mask_ratio=0.5)
    assert loss.requires_grad and mask.shape == (2, 6) and bool(mask.any())

    pix = Seq2Seq(TINY, frontend="pixel")
    warm_start_from_ssl(pix, ssl.encoder_state())
    for a, b in zip(pix.encoder.parameters(), ssl.encoder.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_char_vocab_roundtrip(tmp_path):
    vocab = CharVocab.build(["ⲡⲃⲓⲟⲥ 𓐩𓏌"])
    ids = vocab.encode("ⲡⲃ 𓐩")
    assert vocab.decode(ids) == "ⲡⲃ 𓐩"
    vocab.save(tmp_path / "v.json")
    assert CharVocab.load(tmp_path / "v.json").itos == vocab.itos


def test_sentencepiece_in_repo_roundtrip(tmp_path):
    texts = [f"sentence number {i} with some words" for i in range(60)]
    model = train_sentencepiece(texts, tmp_path / "sp" / "toy", vocab_size=80)
    sub = Subwords(model)
    ids = sub.encode("sentence with words")
    assert ids[0] == 1 and ids[-1] == 2  # BOS/EOS
    assert sub.decode(ids) == "sentence with words"


def test_lr_schedule_warmup_then_decay():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(399, cfg) == pytest.approx(1e-3 * 0.5)  # sqrt(100/400)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = Seq2Seq(TINY, frontend="tokens")
    torch.save({"step": 7, "model": model.state_dict()}, tmp_path / "best.pt")
    clone = Seq2Seq(TINY, frontend="tokens")
    assert load_checkpoint(clone, tmp_path / "best.pt") == 7
    for a, b in zip(model.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_full_size_param_count_in_target_band():
    """Spec target band is 50-400M; the default pixel model must sit in it."""
    full = Seq2Seq(ModelConfig(), frontend="pixel")
    assert 50_000_000 < count_params(full) < 400_000_000


def test_pick_device_explicit():
    assert pick_device("cpu").type == "cpu"
