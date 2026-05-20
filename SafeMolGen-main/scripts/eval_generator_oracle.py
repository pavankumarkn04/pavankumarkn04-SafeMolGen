#!/usr/bin/env python3
"""Evaluate generator(s) with the DrugOracle: overall and phase success %.

Overall probability is calibrated (0–100%) so values can exceed 50%. Compares
base vs RL generator (if present). Reports mean, max, p90 of overall and phases.

  PYTHONPATH=. python scripts/eval_generator_oracle.py
  PYTHONPATH=. python scripts/eval_generator_oracle.py --n 200 --base checkpoints/generator --rl checkpoints/generator_rl
  PYTHONPATH=. python scripts/eval_generator_oracle.py --base checkpoints/generator/best --target-phase 0.6  # Option B: eval with condition
"""

import argparse
import sys
from pathlib import Path

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle
from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles
from utils.condition_vector import get_target_condition


def _stats(values):
    if not values:
        return {"mean": 0, "max": 0, "p90": 0, "p95": 0, "n": 0}
    s = sorted(values, reverse=True)
    n = len(s)
    p90 = s[int(0.90 * n) - 1] if n >= 10 else s[-1]
    p95 = s[int(0.95 * n) - 1] if n >= 20 else s[-1]
    return {
        "mean": sum(s) / n,
        "max": s[0],
        "p90": p90,
        "p95": p95,
        "n": n,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate generator(s) with DrugOracle")
    p.add_argument("--n", type=int, default=150, help="Number of molecules to generate per model")
    p.add_argument("--base", type=str, default="checkpoints/generator", help="Base generator path")
    p.add_argument("--rl", type=str, default="checkpoints/generator_rl", help="RL generator path (skip if missing)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--target-phase", type=float, default=None, help="If set, use this as target condition when generating (for Option B model)")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_path = project_root / args.base
    rl_path = project_root / args.rl
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"

    if not base_path.exists() or not (base_path / "model.pt").exists():
        print(f"Base generator not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not oracle_path.exists() or not admet_path.exists():
        print("Oracle/ADMET checkpoints not found. Train them first.", file=sys.stderr)
        sys.exit(1)

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    print("Loading pipeline (Oracle + base generator)...")
    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(base_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
    )

    condition = None
    if getattr(args, "target_phase", None) is not None and getattr(pipeline.generator.model, "cond_dim", 0) > 0:
        condition = get_target_condition(device="cpu", phase=args.target_phase)
        print(f"Using target condition (phase={args.target_phase}) for generation.\n")

    def run_eval(name: str, gen: SafeMolGen, cond=condition):
        samples = gen.generate(
            n=args.n,
            temperature=args.temperature,
            top_k=args.top_k,
            device="cpu",
            condition=cond,
        )
        valid = [s for s in samples if validate_smiles(s)]
        overalls = []
        phase1s, phase2s, phase3s = [], [], []
        for s in valid:
            pred = pipeline.oracle.predict(s)
            if pred is not None:
                overalls.append(pred.overall_prob * 100)
                phase1s.append(pred.phase1_prob * 100)
                phase2s.append(pred.phase2_prob * 100)
                phase3s.append(pred.phase3_prob * 100)
        validity = len(valid) / max(len(samples), 1) * 100
        uniqueness = len(set(valid)) / max(len(valid), 1) * 100 if valid else 0
        o = _stats(overalls)
        p1 = _stats(phase1s)
        p2 = _stats(phase2s)
        p3 = _stats(phase3s)
        print(f"\n--- {name} ---")
        print(f"  Validity: {len(valid)}/{len(samples)} ({validity:.1f}%)  Uniqueness: {uniqueness:.1f}%")
        print(f"  Oracle-scored: {o['n']} molecules")
        if o["n"] > 0:
            print(f"  Overall %   : mean={o['mean']:.3f}  max={o['max']:.3f}  p90={o['p90']:.3f}  p95={o['p95']:.3f}")
            print(f"  Phase I %   : mean={p1['mean']:.2f}  max={p1['max']:.2f}")
            print(f"  Phase II %  : mean={p2['mean']:.2f}  max={p2['max']:.2f}")
            print(f"  Phase III % : mean={p3['mean']:.2f}  max={p3['max']:.2f}")
        return o

    print(f"\nGenerating {args.n} molecules from BASE generator ({args.base})...")
    o_base = run_eval("Base generator", pipeline.generator, cond=condition)

    if rl_path.exists() and (rl_path / "model.pt").exists():
        print(f"\nLoading RL generator ({args.rl})...")
        gen_rl = SafeMolGen.from_pretrained(str(rl_path), device="cpu")
        print(f"Generating {args.n} molecules from RL generator (same condition as base for fair comparison)...")
        o_rl = run_eval("RL generator", gen_rl, cond=condition)
        if o_base["n"] > 0 and o_rl["n"] > 0:
            print("\n--- Comparison (overall %) ---")
            print(f"  Base: mean={o_base['mean']:.3f}  max={o_base['max']:.3f}")
            print(f"  RL:   mean={o_rl['mean']:.3f}  max={o_rl['max']:.3f}")
            if o_rl["mean"] > o_base["mean"]:
                print("  RL generator has higher mean overall.")
            elif o_rl["max"] > o_base["max"]:
                print("  RL generator has higher max overall.")
            elif o_rl["mean"] < o_base["mean"] * 0.9 or o_rl["max"] < o_base["max"] * 0.9:
                print("  Base beats RL. Re-train RL from Option B: bash scripts/run_rl_from_option_b.sh then re-eval.")
            else:
                print("  RL and base are similar; try scripts/run_rl_curriculum.sh (curriculum RL) or scripts/run_best_of_n_then_eval.sh (Best-of-N).")
    else:
        print(f"\nRL generator not found at {rl_path}; skipping RL eval.")

    print()


if __name__ == "__main__":
    main()
