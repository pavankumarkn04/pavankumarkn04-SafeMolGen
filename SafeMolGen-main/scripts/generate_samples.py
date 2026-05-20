"""Generate SMILES samples from a trained generator (same CLI as main project)."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/generator")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--out", type=str, default="outputs/generator_samples.txt")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not (model_path / "model.pt").exists():
        print(f"Generator checkpoint not found at {model_path}. Use --model /path/to/checkpoint", file=sys.stderr)
        sys.exit(1)

    gen = SafeMolGen.from_pretrained(str(model_path))
    if args.valid_only:
        samples = gen.generate_valid(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_attempts_per_sample=args.max_attempts,
            max_length=args.max_length,
        )
    else:
        samples = gen.generate(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            max_length=args.max_length,
        )

    valid = [s for s in samples if validate_smiles(s)]
    unique = set(samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Generated samples (n={len(samples)})\n")
        for s in samples:
            f.write(s + "\n")
        f.write("\nValid samples\n")
        for s in valid:
            f.write(s + "\n")

    print(f"Generated: {len(samples)}")
    print(f"Valid: {len(valid)} ({len(valid)/max(len(samples), 1):.2%})")
    print(f"Unique: {len(unique)} ({len(unique)/max(len(samples), 1):.2%})")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
