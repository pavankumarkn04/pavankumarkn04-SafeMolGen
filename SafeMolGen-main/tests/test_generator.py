"""Unit tests for SafeMolGen (no checkpoint required)."""

import pytest
import torch

from models.generator.safemolgen import SafeMolGen
from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import CausalTransformerGenerator
from utils.chemistry import validate_smiles


def _make_tiny_generator(seed: int = 42) -> SafeMolGen:
    """Build SafeMolGen with minimal vocab and random weights for testing."""
    torch.manual_seed(seed)
    vocab = {
        SMILESTokenizer.PAD_TOKEN: 0,
        SMILESTokenizer.BOS_TOKEN: 1,
        SMILESTokenizer.EOS_TOKEN: 2,
        SMILESTokenizer.UNK_TOKEN: 3,
        "C": 4, "c": 5, "(": 6, ")": 7, "=": 8, "N": 9, "O": 10,
    }
    tokenizer = SMILESTokenizer(vocab=vocab, max_length=32)
    model = CausalTransformerGenerator(
        vocab_size=len(vocab),
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_len=32,
        cond_dim=0,
    )
    return SafeMolGen(tokenizer, model)


def test_generate_returns_n_strings():
    gen = _make_tiny_generator()
    out = gen.generate(n=5, temperature=0.8, top_k=5, device="cpu")
    assert len(out) == 5
    assert all(isinstance(s, str) for s in out)
    assert all(len(s) >= 0 for s in out)


def test_generate_deterministic_with_temp_zero():
    gen = _make_tiny_generator(seed=123)
    a = gen.generate(n=3, temperature=0.0, top_k=1, device="cpu")
    gen2 = _make_tiny_generator(seed=123)
    b = gen2.generate(n=3, temperature=0.0, top_k=1, device="cpu")
    assert a == b


def test_generate_with_condition_cond_dim_zero():
    gen = _make_tiny_generator()
    # cond_dim=0 model ignores condition; should not raise
    out = gen.generate(n=2, condition=torch.zeros(1, 25), device="cpu")
    assert len(out) == 2


def test_generate_valid_uses_validate_smiles():
    gen = _make_tiny_generator()
    # With random tiny model we may get 0 valid; just check it returns list and doesn't raise
    out = gen.generate_valid(n=3, max_attempts_per_sample=5, device="cpu")
    assert isinstance(out, list)
    for s in out:
        assert validate_smiles(s)


def test_validate_smiles():
    assert validate_smiles("CCO") is True
    assert validate_smiles("c1ccccc1") is True
    assert validate_smiles("invalid!!!") is False
    assert validate_smiles("") is False


def test_tokenizer_encode_decode_roundtrip():
    vocab = {
        SMILESTokenizer.PAD_TOKEN: 0,
        SMILESTokenizer.BOS_TOKEN: 1,
        SMILESTokenizer.EOS_TOKEN: 2,
        SMILESTokenizer.UNK_TOKEN: 3,
        "C": 4, "O": 5, "(": 6, ")": 7,
    }
    tok = SMILESTokenizer(vocab=vocab, max_length=20)
    s = "CCO"
    ids = tok.encode(s)
    decoded = tok.decode(ids)
    assert decoded == s
