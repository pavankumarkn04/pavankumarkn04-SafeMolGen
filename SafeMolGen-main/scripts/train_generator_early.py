#!/usr/bin/env python3
"""
Train a weak (early) generator for iteration 0 only, so the optimization journey
shows visible improvement (weak first batch → stronger later iterations).

Usage (from project root, with venv and PYTHONPATH):
  python3 scripts/train_generator_early.py

This runs pretrain with fewer epochs (default 5) and saves to
checkpoints/generator_early. Then set in config/pipeline.yaml:
  generator_early_path: checkpoints/generator_early
and restart the backend so the pipeline uses it for the first iteration.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = "checkpoints/generator_early"
EPOCHS = 5
LIMIT = 50000
BATCH = 64


def main():
    out_dir = PROJECT_ROOT / OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_generator.py"),
        "--stage", "pretrain",
        "--out", OUT,
        "--epochs", str(EPOCHS),
        "--limit", str(LIMIT),
        "--batch-size", str(BATCH),
    ]
    print("Training weak (early) generator for iteration 0...")
    print("  out:", OUT, "  epochs:", EPOCHS, "  limit:", LIMIT)
    print("  Then set config/pipeline.yaml: generator_early_path:", OUT)
    print()
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)
    best = out_dir / "best" / "model.pt"
    if best.exists():
        print("\nDone. Early generator saved at", out_dir, "(best in best/).")
        print("Uncomment generator_early_path in config/pipeline.yaml and restart the backend.")
    else:
        print("\nNo best checkpoint saved; check logs.")


if __name__ == "__main__":
    main()
