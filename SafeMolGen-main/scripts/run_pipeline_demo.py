"""Run pipeline in demo mode: low-baseline seed + phase-weighted selection for visible improvement.

Uses a fixed seed SMILES that typically scores low (e.g. simple amine), so the optimization
journey can show clear improvement in Phase I/II and overall. Output: outputs/design_demo.json

  PYTHONPATH=. python scripts/run_pipeline_demo.py
  PYTHONPATH=. python scripts/run_pipeline_demo.py --out my_demo.json
"""

from pathlib import Path
import json
import sys

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle, DesignResult


# Seed that often has suboptimal ADMET (simple alkyl amine) so optimization can show improvement
DEMO_BAD_SEED_SMILES = "CCCCCCN"


def _monitor_callback(result: DesignResult) -> None:
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
    generator_path = project_root / "checkpoints" / "generator_best_of_n"
    if not generator_path.exists() or not (generator_path / "model.pt").exists():
        generator_path = project_root / "checkpoints" / "generator" / "best"
    if not generator_path.exists() or not (generator_path / "model.pt").exists():
        generator_path = project_root / "checkpoints" / "generator"
    if not (generator_path / "model.pt").exists():
        print("Generator checkpoint missing. Train models first.", file=sys.stderr)
        sys.exit(1)

    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    if not oracle_path.exists() or not admet_path.exists():
        print("Oracle or ADMET checkpoint missing. Train models first.", file=sys.stderr)
        sys.exit(1)

    endpoints_cfg = yaml.safe_load((project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))

    reranker_path = None
    rp = project_root / "checkpoints" / "reranker" / "reranker.pt"
    if rp.exists():
        reranker_path = str(rp)

    print("Demo: bad seed =", DEMO_BAD_SEED_SMILES, "| selection = phase_weighted | target = 0.35", flush=True)
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

    design_kw = dict(
        target_success=0.35,
        max_iterations=10,
        candidates_per_iteration=100,
        show_progress=True,
        use_oracle_feedback=True,
        on_iteration_done=_monitor_callback,
        selection_mode="phase_weighted",
        use_phase_aware_steering=True,
        exploration_fraction=0.15,
        seed_smiles=DEMO_BAD_SEED_SMILES,
        use_reranker=bool(reranker_path),
        reranker_top_k=200,
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
        smi_short = (result.final_smiles[:70] + "...") if len(result.final_smiles) > 70 else result.final_smiles
        print("Best SMILES:", smi_short, flush=True)

    out_path = project_root / "outputs" / "design_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print("Result saved to", out_path, flush=True)


if __name__ == "__main__":
    main()
