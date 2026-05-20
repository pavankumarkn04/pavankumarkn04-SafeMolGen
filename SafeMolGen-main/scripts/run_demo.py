#!/usr/bin/env python3
"""Generator demo (terminal): load model and generate molecules. Use ./run from repo root."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles

# Formatting
W = 72
SEP = "=" * W
THIN = "-" * W


def _truncate(s: str, max_len: int = 70) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 2] + ".."


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeMolGen generator demo (terminal only)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to generator checkpoint")
    parser.add_argument("--n", type=int, default=20, help="Number of molecules to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--valid-only", action="store_true", help="Only output valid SMILES")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        for candidate in [
            ROOT / "checkpoints" / "generator",
            Path(__file__).resolve().parents[2] / "SafeMolGen-DrugOracle" / "checkpoints" / "generator",
        ]:
            if (candidate / "model.pt").exists() and (candidate / "tokenizer.json").exists():
                model_path = str(candidate)
                break
        if not model_path:
            print("No generator checkpoint found. Place model.pt and tokenizer.json in checkpoints/generator/")
            print("Or: --model $HOME/Documents/Projects/MiniProject/SafeMolGen-DrugOracle/checkpoints/generator")
            sys.exit(1)
    else:
        p = Path(model_path)
        if not (p / "model.pt").exists() or not (p / "tokenizer.json").exists():
            print(f"Checkpoint missing model.pt or tokenizer.json at: {p}")
            sys.exit(1)

    print(SEP)
    print("  SafeMolGen — Generator demo (terminal)")
    print(SEP)
    print(f"  Model:  {_truncate(model_path)}")
    print(f"  Count:  {args.n}  |  Temperature: {args.temperature}  |  top_k: {args.top_k}")
    print(THIN)

    print("  Loading generator...")
    gen = SafeMolGen.from_pretrained(model_path, device=args.device)
    print("  Loaded. Generating...")
    print(THIN)

    if args.valid_only:
        samples = gen.generate_valid(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length,
            device=args.device,
        )
        valid_list = samples
        invalid_list = []
    else:
        samples = gen.generate(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length,
            device=args.device,
        )
        valid_list = [s for s in samples if validate_smiles(s)]
        invalid_list = [s for s in samples if not validate_smiles(s)]

    n_valid = len(valid_list)
    n_invalid = len(invalid_list)
    n_total = len(samples)
    pct = (n_valid / n_total * 100) if n_total else 0

    print("  SUMMARY")
    print(THIN)
    print(f"    Total generated:  {n_total}")
    print(f"    Valid (RDKit):    {n_valid}  ({pct:.1f}%)")
    print(f"    Invalid:          {n_invalid}")
    print(THIN)

    if valid_list:
        print("  VALID SMILES")
        print(THIN)
        for i, s in enumerate(valid_list, 1):
            print(f"    {i:3}. {_truncate(s)}")
        print()

    if invalid_list:
        print("  INVALID SMILES (RDKit parse errors printed above for each)")
        print(THIN)
        for i, s in enumerate(invalid_list, 1):
            print(f"    {i:3}. {_truncate(s)}")
        print()

    print(SEP)
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
