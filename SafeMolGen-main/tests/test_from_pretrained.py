"""Test from_pretrained with a temporary checkpoint (no external checkpoint needed)."""

import tempfile
from pathlib import Path

import torch

from models.generator.safemolgen import SafeMolGen
from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import CausalTransformerGenerator


def test_from_pretrained_roundtrip():
    """Save a minimal generator to a temp dir and load it back; then generate."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        vocab = {
            SMILESTokenizer.PAD_TOKEN: 0,
            SMILESTokenizer.BOS_TOKEN: 1,
            SMILESTokenizer.EOS_TOKEN: 2,
            SMILESTokenizer.UNK_TOKEN: 3,
            "C": 4, "O": 5, "(": 6, ")": 7,
        }
        tokenizer = SMILESTokenizer(vocab=vocab, max_length=32)
        tokenizer.save(root / "tokenizer.json")
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
        torch.save({
            "model": model.state_dict(),
            "config": {"cond_dim": 0, "d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1},
        }, root / "model.pt")

        gen = SafeMolGen.from_pretrained(str(root), device="cpu")
        out = gen.generate(n=3, temperature=0.5, top_k=4, device="cpu")
        assert len(out) == 3
        assert all(isinstance(s, str) for s in out)
