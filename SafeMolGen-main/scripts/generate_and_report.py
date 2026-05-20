#!/usr/bin/env python3
"""Generate N SMILES with prj_demo generator and report validity/uniqueness. No server required.
Usage: PYTHONPATH=. python scripts/generate_and_report.py [--n 1000]
Note: The generator does not compute loss at inference; only validity and uniqueness are reported.
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles
from utils.condition_vector import get_target_condition


def _get_checkpoint_path() -> Path:
    import os
    env_path = os.environ.get("GENERATOR_CHECKPOINT")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    default = ROOT / "checkpoints" / "generator"
    if (default / "model.pt").exists():
        return default
    fallback = Path.home() / "Documents" / "Projects" / "MiniProject" / "SafeMolGen-DrugOracle" / "checkpoints" / "generator"
    if (fallback / "model.pt").exists():
        return fallback
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    path = _get_checkpoint_path()
    if not (path / "model.pt").exists() or not (path / "tokenizer.json").exists():
        print(f"Generator not found at {path}", file=sys.stderr)
        sys.exit(2)

    gen = SafeMolGen.from_pretrained(str(path), device="cpu")
    cond_dim = getattr(gen.model, "cond_dim", 0)
    condition = get_target_condition(device="cpu", phase=0.6) if cond_dim > 0 else None

    print(f"Generating {args.n} SMILES (temp={args.temperature}, top_k={args.top_k})...")
    samples = gen.generate(
        n=args.n,
        temperature=args.temperature,
        top_k=args.top_k,
        device="cpu",
        condition=condition,
    )
    valid = sum(1 for s in samples if validate_smiles(s))
    unique = len(set(samples))
    total = len(samples)
    validity_pct = 100.0 * valid / total if total else 0
    uniqueness_pct = 100.0 * unique / total if total else 0

    print(f"Total:   {total}")
    print(f"Valid:   {valid} ({validity_pct:.2f}%)")
    print(f"Unique:  {unique} ({uniqueness_pct:.2f}%)")
    print("(No loss at inference; generator only reports validity/uniqueness.)")


if __name__ == "__main__":
    main()
