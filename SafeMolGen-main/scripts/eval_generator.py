"""Evaluate generator checkpoint: report validity and uniqueness; exit non-zero if below thresholds.

Run from project root: PYTHONPATH=. python scripts/eval_generator.py --model checkpoints/generator
"""

import argparse
import sys
from pathlib import Path

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/generator")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--validity-min", type=float, default=0.85, help="Min validity fraction (default 0.85)")
    parser.add_argument("--uniqueness-min", type=float, default=0.90, help="Min uniqueness fraction (default 0.90)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not (model_path / "model.pt").exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    gen = SafeMolGen.from_pretrained(str(model_path), device="cpu")
    samples = gen.generate(n=args.n, temperature=args.temperature, top_k=args.top_k, device="cpu")
    valid = sum(1 for s in samples if validate_smiles(s))
    unique = len(set(samples))
    validity = valid / max(len(samples), 1)
    uniqueness = unique / max(len(samples), 1)

    print(f"Validity: {valid}/{len(samples)} ({validity:.1%})")
    print(f"Uniqueness: {unique}/{len(samples)} ({uniqueness:.1%})")
    if validity < args.validity_min or uniqueness < args.uniqueness_min:
        print(f"Below thresholds (validity>={args.validity_min}, uniqueness>={args.uniqueness_min})", file=sys.stderr)
        sys.exit(1)
    print("Passed thresholds.")


if __name__ == "__main__":
    main()
