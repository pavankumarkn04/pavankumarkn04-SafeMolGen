#!/usr/bin/env python3
"""Collect (condition_vector, best_smiles) from design_molecule for imitation learning (plan 3.2).

  PYTHONPATH=. python3 scripts/collect_imitation_data.py --runs 50 --out data/processed/generator/imitation_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle
from utils.chemistry import validate_smiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=50, help="Number of design_molecule runs")
    p.add_argument("--out", type=str, default="data/processed/generator/imitation_pairs.jsonl")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--candidates-per-iteration", type=int, default=100)
    p.add_argument("--show-progress", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator_path = root / "checkpoints" / "generator" / "best"
    if not (generator_path / "model.pt").exists():
        generator_path = root / "checkpoints" / "generator"
    oracle_path = root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = root / "config" / "endpoints.yaml"
    if not generator_path.exists() or not (generator_path / "model.pt").exists():
        print("Generator checkpoint not found.", file=sys.stderr)
        sys.exit(1)
    if not oracle_path.exists() or not admet_path.exists():
        print("Oracle/ADMET not found.", file=sys.stderr)
        sys.exit(1)

    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    print("Loading pipeline...")
    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
    )

    def imitation_cb(_result, condition, best_smiles: str):
        if not best_smiles or not validate_smiles(best_smiles):
            return
        if condition is None:
            return
        cond_list = condition.flatten().tolist()
        f.write(json.dumps({"condition": cond_list, "smiles": best_smiles}) + "\n")
        f.flush()

    with open(out_path, "w", encoding="utf-8") as f:
        for run in range(args.runs):
            if args.show_progress:
                print(f"Run {run + 1}/{args.runs} ...")
            pipeline.design_molecule(
                max_iterations=args.max_iterations,
                candidates_per_iteration=args.candidates_per_iteration,
                show_progress=False,
                imitation_callback=imitation_cb,
            )
    print(f"Collected from {args.runs} runs. Use scripts/train_imitation.py to train.")


if __name__ == "__main__":
    main()
