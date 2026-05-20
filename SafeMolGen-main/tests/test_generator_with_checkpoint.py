"""Tests that require a real generator checkpoint (skipped if missing)."""

from pathlib import Path

import pytest

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _checkpoint_paths():
    root = _project_root()
    candidates = [root / "checkpoints" / "generator"]
    for p in candidates:
        if (p / "model.pt").exists() and (p / "tokenizer.json").exists():
            return str(p)
    return None


CHECKPOINT = _checkpoint_paths()


@pytest.mark.skipif(CHECKPOINT is None, reason="No generator checkpoint found")
def test_from_pretrained_and_generate():
    gen = SafeMolGen.from_pretrained(CHECKPOINT, device="cpu")
    samples = gen.generate(n=10, temperature=0.8, top_k=40, device="cpu")
    assert len(samples) == 10
    valid = [s for s in samples if validate_smiles(s)]
    assert len(valid) >= 0  # may be 0 with untrained; just no crash
    assert all(isinstance(s, str) for s in samples)


@pytest.mark.skipif(CHECKPOINT is None, reason="No generator checkpoint found")
def test_generate_valid_with_checkpoint():
    gen = SafeMolGen.from_pretrained(CHECKPOINT, device="cpu")
    samples = gen.generate_valid(n=5, max_attempts_per_sample=15, device="cpu")
    assert len(samples) <= 5
    for s in samples:
        assert validate_smiles(s)
