#!/usr/bin/env python3
"""Option B with target-condition curriculum (plan 3.3): pretrain 0.5 -> 0.55 -> 0.6.

Teaches the generator to satisfy increasingly strict targets without RL.
  PYTHONPATH=. python scripts/run_option_b_curriculum.py
  PYTHONPATH=. python scripts/run_option_b_curriculum.py --phases 0.5,0.55,0.6 --epochs-per-stage 10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Option B curriculum: pretrain with increasing target phase")
    p.add_argument("--curated-tsv", default="data/processed/generator/smiles_oracle_curated.tsv")
    p.add_argument("--phases", type=str, default="0.5,0.55,0.6", help="Comma-separated target phases (e.g. 0.5,0.55,0.6)")
    p.add_argument("--top-pct", type=float, default=20.0)
    p.add_argument("--min-overall", type=float, default=None, help="Keep molecules with overall_prob >= this (plan 4.2)")
    p.add_argument("--limit", type=int, default=200_000)
    p.add_argument("--epochs-per-stage", type=int, default=10, help="Pretrain epochs per curriculum stage")
    p.add_argument("--out", default="checkpoints/generator", help="Generator output dir")
    p.add_argument("--skip-curate", action="store_true", help="Skip curate step (use existing TSV)")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    curated = root / args.curated_tsv if not Path(args.curated_tsv).is_absolute() else Path(args.curated_tsv)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    phases = [float(x.strip()) for x in args.phases.split(",")]

    def run(cmd: list) -> None:
        print(f"\n>> {' '.join(cmd)}\n")
        r = subprocess.run(cmd, cwd=str(root), env={**os.environ, "PYTHONPATH": str(root)})
        if r.returncode != 0:
            sys.exit(r.returncode)

    if not args.skip_curate:
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
    else:
        if not curated.exists():
            print(f"Curated TSV not found: {curated}. Run without --skip-curate first.", file=sys.stderr)
            sys.exit(1)

    print("=== Step 2: Target-condition curriculum pretrain ===")
    for i, phase in enumerate(phases):
        print(f"--- Curriculum stage {i + 1}/{len(phases)}: target_phase={phase} ---")
        cmd = [
            sys.executable,
            "scripts/train_generator.py",
            "--stage", "pretrain",
            "--data", str(curated),
            "--use-target-condition",
            "--target-phase", str(phase),
            "--limit", str(args.limit),
            "--epochs", str(args.epochs_per_stage),
            "--out", str(out_dir),
        ]
        if i > 0:
            cmd.extend(["--resume", str(out_dir)])
        run(cmd)

    print("\nDone. Generator:", str(out_dir / "best" if (out_dir / "best" / "model.pt").exists() else out_dir))
    print("Eval: PYTHONPATH=. python scripts/eval_generator_oracle.py --base", str(out_dir / "best"), "--target-phase", str(phases[-1]), "--n 150")


if __name__ == "__main__":
    main()
