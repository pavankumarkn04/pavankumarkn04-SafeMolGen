"""Run design_molecule with live CLI monitoring and a final per-iteration table.

  PYTHONPATH=. python scripts/run_design_and_monitor.py
  PYTHONPATH=. python scripts/run_design_and_monitor.py --max-iterations 5 --safety-threshold 0.02
"""

from pathlib import Path
import argparse
import sys

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle, DesignResult


def _monitor_callback(result: DesignResult) -> None:
    """Print one line per iteration as it completes."""
    if not result.iteration_history:
        return
    r = result.iteration_history[-1]
    p = r.prediction
    smi = (r.smiles[:45] + "…") if len(r.smiles) > 45 else r.smiles
    print(
        "  iter {:2d}  overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  |  {}".format(
            r.iteration, p.overall_prob, p.phase1_prob, p.phase2_prob, p.phase3_prob, smi
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run design_molecule with live CLI monitoring"
    )
    parser.add_argument("--target-success", type=float, default=0.25, help="Target overall success probability")
    parser.add_argument("--safety-threshold", type=float, default=0.02, help="Minimum safety bar")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max design iterations")
    parser.add_argument("--candidates-per-iteration", type=int, default=100, help="Candidates per iteration")
    parser.add_argument("--use-rl-model", action="store_true", help="Use checkpoints/generator_rl if available")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    generator_path = project_root / "checkpoints" / "generator_rl" if args.use_rl_model else project_root / "checkpoints" / "generator"
    if not generator_path.exists():
        generator_path = project_root / "checkpoints" / "generator"
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"

    if not generator_path.exists() or not oracle_path.exists() or not admet_path.exists():
        print("Missing checkpoints. Train ADMET, Oracle, and Generator first.", file=sys.stderr)
        sys.exit(1)

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

    print("Running design_molecule (target_success={}, safety_threshold={}, max_iterations={}) ...".format(
        args.target_success, args.safety_threshold, args.max_iterations,
    ), flush=True)
    print("Live monitor:", flush=True)
    result = pipeline.design_molecule(
        target_success=args.target_success,
        max_iterations=args.max_iterations,
        candidates_per_iteration=args.candidates_per_iteration,
        show_progress=True,
        safety_threshold=args.safety_threshold,
        use_oracle_feedback=True,
        on_iteration_done=_monitor_callback,
    )

    print("", flush=True)
    print("=" * 80)
    print("PER-ITERATION SUMMARY")
    print("=" * 80)
    for i, r in enumerate(result.iteration_history):
        pred = r.prediction
        smiles_short = (r.smiles[:50] + "…") if len(r.smiles) > 50 else r.smiles
        feedback_used_next = (
            result.iteration_history[i + 1].used_oracle_feedback
            if i + 1 < len(result.iteration_history) else False
        )
        print(
            "Iter {:2d}  overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  "
            "passed_safety={}  used_feedback={}  feedback_for_next={}".format(
                r.iteration,
                pred.overall_prob,
                pred.phase1_prob,
                pred.phase2_prob,
                pred.phase3_prob,
                r.passed_safety,
                r.used_oracle_feedback,
                feedback_used_next,
            )
        )
        print("       SMILES: {}".format(smiles_short))
    print("=" * 80)
    print(
        "Final best: overall={:.2%}  target_achieved={}  total_iterations={}".format(
            result.final_prediction.overall_prob,
            result.target_achieved,
            result.total_iterations,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
