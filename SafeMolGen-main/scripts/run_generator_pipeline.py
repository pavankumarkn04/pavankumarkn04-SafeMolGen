#!/usr/bin/env python3
"""
Run generator pipeline in order: data check -> pretrain -> validity check -> optional RL.
Execute from project root with PYTHONPATH=. and deps installed (torch, rdkit, etc.).

  PYTHONPATH=. python3 scripts/run_generator_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    p = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )
    if p.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(p.returncode)


def main() -> None:
    data_path = PROJECT_ROOT / "data/processed/generator/smiles.tsv"
    admet_base = PROJECT_ROOT / "data/admet_group"
    if not data_path.exists() and not (admet_base.exists() and list(admet_base.glob("*/train_val.csv"))):
        print("No generator data. Run: python3 scripts/download_chembl_smiles.py")
        print("Or ensure data/admet_group/*/train_val.csv exist.")
        sys.exit(1)
    if data_path.exists():
        lines = len(data_path.read_text().splitlines())
        print(f"Generator data: {data_path} ({lines} lines)")
    else:
        print("Using ADMET fallback for SMILES.")

    run(
        [sys.executable, "scripts/train_generator.py", "--stage", "pretrain", "--epochs", "3", "--limit", "20000", "--batch-size", "64"],
        "Step 1: Pretrain (3 epochs, 20k SMILES)",
    )
    run(
        [sys.executable, "scripts/generate_samples.py", "--model", "checkpoints/generator", "--n", "100", "--top-k", "40"],
        "Step 2: Validity check (100 samples)",
    )
    print("\nIf validity > 0%, run RL: PYTHONPATH=. python3 scripts/train_generator.py --stage rl --resume checkpoints/generator")
    print("Done.")


if __name__ == "__main__":
    main()
