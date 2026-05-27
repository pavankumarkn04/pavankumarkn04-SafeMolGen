#!/usr/bin/env python3
"""Run Option B full pipeline: curate -> target-condition pretrain (0.6) -> evaluate.

  PYTHONPATH=. python scripts/run_option_b_full.py
  PYTHONPATH=. python scripts/run_option_b_full.py --epochs 20 --top-pct 15
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Option B: curate + target-condition pretrain + eval")
    p.add_argument("--curated-tsv", default="data/processed/generator/smiles_oracle_curated.tsv", help="Output path for curated SMILES")
    p.add_argument("--target-phase", type=float, default=0.6, help="Success target for all three phases (default 0.6)")
    p.add_argument("--top-pct", type=float, default=20.0, help="Keep top this percent by overall_prob when curating")
    p.add_argument("--min-overall", type=float, default=None, help="Alternatively: keep molecules with overall_prob >= this (plan 4.2)")
    p.add_argument("--limit", type=int, default=200_000, help="Max SMILES to score/use")
    p.add_argument("--epochs", type=int, default=30, help="Pretrain epochs")
    p.add_argument("--eval-n", type=int, default=150, help="Number of molecules for eval")
    p.add_argument("--out", default="checkpoints/generator", help="Generator checkpoint output dir")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    curated = root / args.curated_tsv if not Path(args.curated_tsv).is_absolute() else Path(args.curated_tsv)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    def run(cmd: list) -> None:
        print(f"\n>> {' '.join(cmd)}\n")
        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            sys.exit(r.returncode)

    # 1. Build Oracle-curated SMILES
    print("=== Step 1: Build Oracle-curated SMILES ===")
    curate_cmd = [
        sys.executable,
        "scripts/build_oracle_curated_smiles.py",
        "--data", "data/processed/generator/smiles.tsv",
        "--out", str(curated),
        "--limit", str(args.limit),
        "--top-pct", str(args.top_pct),
    ]
    if getattr(args, "min_overall", None) is not None:
        curate_cmd.extend(["--min-overall", str(args.min_overall)])
    run(curate_cmd)

    # 2. Target-condition pretrain with 0.6 success target
    print("=== Step 2: Target-condition pretrain (target_phase=0.6) ===")
    run([
        sys.executable,
        "scripts/train_generator.py",
        "--stage", "pretrain",
        "--data", str(curated),
        "--use-target-condition",
        "--target-phase", str(args.target_phase),
        "--limit", str(args.limit),
        "--epochs", str(args.epochs),
        "--out", str(out_dir),
    ])

    # 3. Evaluate
    print("=== Step 3: Evaluate generator with Oracle ===")
    base = str(out_dir / "best" if (out_dir / "best" / "model.pt").exists() else out_dir)
    run([
        sys.executable,
        "scripts/eval_generator_oracle.py",
        "--n", str(args.eval_n),
        "--base", base,
    ])

    print("\nDone. Generator:", base)
    print("\n--- What now ---")
    print("1. App already uses checkpoints/generator/best when present (Option B best).")
    print("2. Eval with target condition (how Option B was trained):")
    print(f"   python scripts/eval_generator_oracle.py --base {base} --target-phase {args.target_phase} --n 150")
    print("3. Run design in the UI or: python scripts/run_pipeline.py (uses pipeline with this generator).")


if __name__ == "__main__":
    main()
