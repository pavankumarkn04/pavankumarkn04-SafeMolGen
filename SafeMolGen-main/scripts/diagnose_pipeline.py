"""Diagnose design_molecule iteration behavior: log per-iteration scores and conditioning.

Run from project root with PYTHONPATH=.

  PYTHONPATH=. python scripts/diagnose_pipeline.py
  PYTHONPATH=. python scripts/diagnose_pipeline.py --max-iterations 5 --candidates 50

Use this to verify that success probability improves over iterations after training
the generator with real condition vectors.
"""

from pathlib import Path
import argparse
import json
import sys

import torch
import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle


def get_generator_cond_dim(project_root: Path, use_rl: bool) -> int:
    """Read cond_dim from generator checkpoint config."""
    gen_path = project_root / "checkpoints" / ("generator_rl" if use_rl else "generator")
    pt = gen_path / "model.pt"
    if not pt.exists():
        return -1
    try:
        state = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = state.get("config", {})
        return int(cfg.get("cond_dim", 0))
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pipeline iteration and conditioning")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--target-success", type=float, default=0.25)
    parser.add_argument("--use-rl-model", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON path to write result")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    generator_path = project_root / "checkpoints" / ("generator_rl" if args.use_rl_model else "generator")
    if not generator_path.exists():
        generator_path = project_root / "checkpoints" / "generator"
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"

    if not generator_path.exists() or not oracle_path.exists() or not admet_path.exists():
        print("Missing checkpoints. Train ADMET, Oracle, and Generator first.", file=sys.stderr)
        sys.exit(1)

    cond_dim = get_generator_cond_dim(project_root, args.use_rl_model)
    print(f"Generator cond_dim: {cond_dim} (0 = conditioning disabled, 25 = conditioning enabled)")

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
    )

    print(f"\nRunning design_molecule(max_iterations={args.max_iterations}, candidates_per_iteration={args.candidates})")
    result = pipeline.design_molecule(
        target_success=args.target_success,
        max_iterations=args.max_iterations,
        candidates_per_iteration=args.candidates,
        show_progress=True,
        use_oracle_feedback=True,
    )

    print("\n--- Per-iteration log ---")
    log = []
    for r in result.iteration_history:
        overall = r.prediction.overall_prob
        used_fb = r.used_oracle_feedback
        smi_short = (r.smiles[:50] + "…") if len(r.smiles) > 50 else r.smiles
        line = f"  iter {r.iteration}: overall_prob={overall:.2%} used_oracle_feedback={used_fb} smiles={smi_short}"
        print(line)
        log.append({
            "iteration": r.iteration,
            "overall_prob": overall,
            "used_oracle_feedback": used_fb,
            "smiles": r.smiles,
        })

    print(f"\nBest overall: {result.final_prediction.overall_prob:.2%} | target_achieved={result.target_achieved}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"cond_dim": cond_dim, "iterations": log, "final_overall": result.final_prediction.overall_prob}, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
