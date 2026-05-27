#!/usr/bin/env python3
"""Print raw per-iteration probability scores from the pipeline (for verifying if scores change).
Run: PYTHONPATH=. python scripts/check_scores.py --iters 3 --candidates 40
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import yaml
from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--candidates", type=int, default=40)
    args = p.parse_args()

    gen_path = project_root / "checkpoints" / "generator"
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    if not gen_path.exists() or not oracle_path.exists() or not admet_path.exists():
        print("Missing checkpoints", file=sys.stderr)
        sys.exit(1)

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(gen_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=[e.name for e in endpoints],
        endpoint_task_types={e.name: e.task_type for e in endpoints},
        admet_input_dim=get_admet_node_feature_dim(str(admet_path)),
        device="cpu",
    )

    print("Running", args.iters, "iterations,", args.candidates, "candidates...")
    result = pipeline.design_molecule(
        max_iterations=args.iters,
        candidates_per_iteration=args.candidates,
        show_progress=False,
        use_oracle_feedback=True,
    )
    print("\n--- Raw probability scores (backend) ---")
    for r in result.iteration_history:
        p = r.prediction
        print(
            "  iter {:2d}  overall={:.4f}  phase1={:.4f}  phase2={:.4f}  phase3={:.4f}".format(
                r.iteration, p.overall_prob, p.phase1_prob, p.phase2_prob, p.phase3_prob
            )
        )
    print("  (If overall is identical across iters, oracle returns same score; UI applies display trend.)")


if __name__ == "__main__":
    main()
