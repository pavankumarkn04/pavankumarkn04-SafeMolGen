"""Run integrated SafeMolGen-DrugOracle pipeline from CLI with live monitoring.

Loads generator, Oracle, and ADMET checkpoints; runs design_molecule; prints
per-iteration progress to stdout; optionally saves result to JSON.

  PYTHONPATH=. python scripts/run_pipeline.py --out results/design_result.json
  PYTHONPATH=. python scripts/run_pipeline.py                    # monitor only, no JSON
  PYTHONPATH=. python scripts/run_pipeline.py --max-iterations 5 --candidates-per-iteration 80
"""

from pathlib import Path
import argparse
import json
import sys

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle, DesignResult


def _monitor_callback(result: DesignResult) -> None:
    """Print one line per iteration as it completes (for CLI monitoring)."""
    if not result.iteration_history:
        return
    r = result.iteration_history[-1]
    p = r.prediction
    smi = (r.smiles[:45] + "…") if len(r.smiles) > 45 else r.smiles
    print(
        "  iter {:2d}  overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  |  {}".format(
            r.iteration,
            p.overall_prob,
            p.phase1_prob,
            p.phase2_prob,
            p.phase3_prob,
            smi,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SafeMolGen-DrugOracle design pipeline with live CLI monitoring"
    )
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (omit to run without saving)")
    parser.add_argument("--target-success", type=float, default=0.7, help="Target overall success probability (default 0.7 = 70%%)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max design iterations")
    parser.add_argument("--candidates-per-iteration", type=int, default=100, help="Candidates per iteration")
    parser.add_argument("--use-rl-model", action="store_true", help="Use checkpoints/generator_rl if available")
    parser.add_argument("--generator", type=str, default=None, help="Generator checkpoint path (e.g. checkpoints/generator_best_of_n or checkpoints/generator/best)")
    # Plan 5.1: restarts / evolutionary
    parser.add_argument("--restarts", type=int, default=0, help="Run design with N restarts; return best (0 = single run)")
    parser.add_argument("--evolutionary", action="store_true", help="Use evolutionary search instead of iterative design")
    parser.add_argument("--population-size", type=int, default=20, help="Evolutionary: population size")
    parser.add_argument("--generations", type=int, default=10, help="Evolutionary: number of generations")
    # Plan 5.2: selection mode
    parser.add_argument("--selection-mode", choices=["overall", "pareto", "diversity", "phase_weighted", "bottleneck"], default="overall", help="How to pick best per iteration: overall, pareto, diversity, phase_weighted, bottleneck")
    parser.add_argument("--diversity-tanimoto-max", type=float, default=0.7, help="Max Tanimoto to current best for diversity mode")
    # Plan 4.3: reranker
    parser.add_argument("--use-reranker", action="store_true", help="Use two-stage reranker to pre-filter candidates (if checkpoint exists)")
    parser.add_argument("--reranker-path", type=str, default=None, help="Reranker checkpoint dir (default: checkpoints/reranker)")
    parser.add_argument("--reranker-top-k", type=int, default=200, help="Keep top K candidates after reranking")
    parser.add_argument("--all-solutions", action="store_true", help="Enable restarts=5, selection-mode=diversity, use-reranker when available")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.generator:
        generator_path = Path(args.generator)
        if not generator_path.is_absolute():
            generator_path = project_root / generator_path
        if not (generator_path / "model.pt").exists():
            print("Generator checkpoint missing model.pt at {}.".format(generator_path), file=sys.stderr)
            sys.exit(1)
    elif args.use_rl_model:
        generator_path = project_root / "checkpoints" / "generator_rl"
        if (generator_path / "best" / "model.pt").exists():
            generator_path = generator_path / "best"
    else:
        # Prefer Best-of-N (improved) when available, else Option B best (same as backend)
        best_of_n = project_root / "checkpoints" / "generator_best_of_n"
        if best_of_n.exists() and (best_of_n / "model.pt").exists():
            generator_path = best_of_n
        else:
            generator_path = project_root / "checkpoints" / "generator" / "best"
            if not generator_path.exists() or not (generator_path / "model.pt").exists():
                generator_path = project_root / "checkpoints" / "generator"
    if not args.generator and (not generator_path.exists() or not (generator_path / "model.pt").exists()):
        generator_path = project_root / "checkpoints" / "generator"
    if not (generator_path / "model.pt").exists():
        print("Generator checkpoint missing model.pt at {}.".format(generator_path), file=sys.stderr)
        sys.exit(1)
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"

    if not generator_path.exists() or not oracle_path.exists() or not admet_path.exists():
        print("Missing checkpoints. Train ADMET, Oracle, and Generator first.", file=sys.stderr)
        sys.exit(1)

    # Apply --all-solutions: restarts, diversity selection, reranker when available
    restarts = args.restarts
    selection_mode = args.selection_mode
    use_reranker = args.use_reranker
    if args.all_solutions:
        restarts = max(restarts, 5)
        selection_mode = "diversity"
        use_reranker = True

    reranker_path = None
    if use_reranker:
        rp = args.reranker_path or "checkpoints/reranker"
        rp_path = project_root / rp if not Path(rp).is_absolute() else Path(rp)
        candidate = (rp_path / "reranker.pt") if rp_path.is_dir() else rp_path
        if candidate.exists():
            reranker_path = str(candidate)
        if use_reranker and not reranker_path:
            use_reranker = False
            if args.all_solutions:
                pass  # all-solutions: reranker optional
            elif args.use_reranker:
                print("Reranker path not found; running without reranker.", file=sys.stderr)

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    try:
        gen_label = generator_path.relative_to(project_root)
    except ValueError:
        gen_label = generator_path
    print("Loading pipeline (generator={}) ...".format(gen_label), flush=True)
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
    if reranker_path:
        print("Reranker loaded (pre-filter candidates).", flush=True)

    design_kw = dict(
        target_success=args.target_success,
        max_iterations=args.max_iterations,
        candidates_per_iteration=args.candidates_per_iteration,
        show_progress=True,
        use_oracle_feedback=True,
        on_iteration_done=_monitor_callback,
        selection_mode=selection_mode,
        diversity_tanimoto_max=args.diversity_tanimoto_max,
        use_reranker=use_reranker,
        reranker_top_k=args.reranker_top_k,
    )

    if args.evolutionary:
        print(
            "Running design_molecule_evolutionary(population_size={}, generations={})".format(
                args.population_size, args.generations
            ),
            flush=True,
        )
        result = pipeline.design_molecule_evolutionary(
            population_size=args.population_size,
            generations=args.generations,
            target_success=args.target_success,
            show_progress=True,
        )
    elif restarts > 0:
        print(
            "Running design_molecule_with_restarts(n_restarts={}, max_iterations={}, selection_mode={})".format(
                restarts, args.max_iterations, selection_mode
            ),
            flush=True,
        )
        kw_restarts = {k: v for k, v in design_kw.items() if k != "show_progress"}
        result = pipeline.design_molecule_with_restarts(
            n_restarts=restarts,
            show_progress=True,
            **kw_restarts,
        )
    else:
        print(
            "Running design_molecule(max_iterations={}, candidates_per_iteration={}, target_success={}, selection_mode={})".format(
                args.max_iterations, args.candidates_per_iteration, args.target_success, selection_mode
            ),
            flush=True,
        )
        print("Monitor (live per-iteration):", flush=True)
        result = pipeline.design_molecule(**design_kw)

    print("", flush=True)
    print(
        "Final: overall={:.2%}  phase1={:.2%}  phase2={:.2%}  phase3={:.2%}  target_achieved={}".format(
            result.final_prediction.overall_prob,
            result.final_prediction.phase1_prob,
            result.final_prediction.phase2_prob,
            result.final_prediction.phase3_prob,
            result.target_achieved,
        ),
        flush=True,
    )
    if result.final_smiles:
        smi_short = (result.final_smiles[:70] + "…") if len(result.final_smiles) > 70 else result.final_smiles
        print("Best SMILES: {}".format(smi_short), flush=True)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print("Result saved to {}".format(out_path), flush=True)
    else:
        print("(No --out path; result not saved. Use --out outputs/design_result.json to save.)", flush=True)


if __name__ == "__main__":
    main()
