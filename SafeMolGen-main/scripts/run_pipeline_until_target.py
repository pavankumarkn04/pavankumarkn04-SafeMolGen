#!/usr/bin/env python3
"""Run pipeline with escalation until target is achieved or all strategies exhausted.

If the first run (single design_molecule) does not achieve target_success, tries in order:
  1. design_molecule_with_restarts(n_restarts=5)
  2. all-solutions (restarts=5, diversity, reranker)
  3. design_molecule_evolutionary()

Saves the best result (by overall %) and reports which strategy achieved it.
  PYTHONPATH=. python3 scripts/run_pipeline_until_target.py --target-success 0.7 --out outputs/design_until_target.json
"""

from pathlib import Path
import json
import sys

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle, DesignResult


def _monitor(result: DesignResult) -> None:
    if not result.iteration_history:
        return
    r = result.iteration_history[-1]
    p = r.prediction
    smi = (r.smiles[:45] + "...") if len(r.smiles) > 45 else r.smiles
    print(
        "  iter {:2d}  overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  |  {}".format(
            r.iteration, p.overall_prob, p.phase1_prob, p.phase2_prob, p.phase3_prob, smi
        ),
        flush=True,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    generator_path = project_root / "checkpoints" / "generator_best_of_n"
    if not generator_path.exists() or not (generator_path / "model.pt").exists():
        generator_path = project_root / "checkpoints" / "generator" / "best"
    if not generator_path.exists() or not (generator_path / "model.pt").exists():
        generator_path = project_root / "checkpoints" / "generator"
    if not (generator_path / "model.pt").exists():
        print("Generator checkpoint missing.", file=sys.stderr)
        sys.exit(1)

    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    if not oracle_path.exists() or not admet_path.exists():
        print("Oracle or ADMET checkpoint missing.", file=sys.stderr)
        sys.exit(1)

    target_success = 0.7
    max_iterations = 12
    candidates_per_iteration = 100
    out_path = project_root / "outputs" / "design_until_target.json"
    if "--target-success" in sys.argv:
        i = sys.argv.index("--target-success")
        if i + 1 < len(sys.argv):
            target_success = float(sys.argv[i + 1])
    if "--max-iterations" in sys.argv:
        i = sys.argv.index("--max-iterations")
        if i + 1 < len(sys.argv):
            max_iterations = int(sys.argv[i + 1])
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        if i + 1 < len(sys.argv):
            out_path = Path(sys.argv[i + 1])
            if not out_path.is_absolute():
                out_path = project_root / out_path

    reranker_path = None
    rp = project_root / "checkpoints" / "reranker" / "reranker.pt"
    if rp.exists():
        reranker_path = str(rp)

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    print("Loading pipeline ...", flush=True)
    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
        reranker_path=reranker_path,
    )

    base_kw = dict(
        target_success=target_success,
        max_iterations=max_iterations,
        candidates_per_iteration=candidates_per_iteration,
        show_progress=True,
        use_oracle_feedback=True,
        on_iteration_done=_monitor,
        use_reranker=bool(reranker_path),
        reranker_top_k=200,
    )

    best_result = None
    best_strategy = None

    def _update_best(res: DesignResult, strategy: str) -> None:
        nonlocal best_result, best_strategy
        if best_result is None or res.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
            best_result = res
            best_strategy = strategy

    # Strategy 1: single run (phase_weighted)
    print("\n=== Strategy 1: single run (phase_weighted) ===", flush=True)
    result = pipeline.design_molecule(selection_mode="phase_weighted", **base_kw)
    _update_best(result, "single_phase_weighted")
    print("Result: {:.2%}  target_achieved={}".format(result.final_prediction.overall_prob, result.target_achieved), flush=True)

    # Strategy 2: restarts=5
    print("\n=== Strategy 2: restarts=5 ===", flush=True)
    kw = {k: v for k, v in base_kw.items() if k != "show_progress"}
    result = pipeline.design_molecule_with_restarts(n_restarts=5, show_progress=True, selection_mode="phase_weighted", **kw)
    _update_best(result, "restarts_5")
    print("Result: {:.2%}  target_achieved={}".format(result.final_prediction.overall_prob, result.target_achieved), flush=True)

    # Strategy 3: all-solutions (restarts=5, diversity, reranker)
    print("\n=== Strategy 3: all-solutions (restarts=5, diversity, reranker) ===", flush=True)
    result = pipeline.design_molecule_with_restarts(
        n_restarts=5, show_progress=True, selection_mode="diversity", use_reranker=bool(reranker_path), **kw
    )
    _update_best(result, "all_solutions")
    print("Result: {:.2%}  target_achieved={}".format(result.final_prediction.overall_prob, result.target_achieved), flush=True)

    # Strategy 4: evolutionary
    print("\n=== Strategy 4: evolutionary ===", flush=True)
    result = pipeline.design_molecule_evolutionary(
        population_size=25, generations=12, target_success=target_success, show_progress=True
    )
    _update_best(result, "evolutionary")
    print("Result: {:.2%}  target_achieved={}".format(result.final_prediction.overall_prob, result.target_achieved), flush=True)

    if best_result is None:
        print("No result.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = best_result.to_dict()
    out_data["_strategy"] = best_strategy
    out_data["_target_requested"] = target_success
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print("", flush=True)
    print(
        "Best: overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  strategy={}  target_achieved={}".format(
            best_result.final_prediction.overall_prob,
            best_result.final_prediction.phase1_prob,
            best_result.final_prediction.phase2_prob,
            best_result.final_prediction.phase3_prob,
            best_strategy,
            best_result.target_achieved,
        ),
        flush=True,
    )
    print("Result saved to {}".format(out_path), flush=True)


if __name__ == "__main__":
    main()
