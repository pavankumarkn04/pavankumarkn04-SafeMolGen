#!/usr/bin/env python3
"""Build (condition, smiles, oracle_score) dataset for reranker training (plan 4.3).

Samples SMILES from the generator under various conditions and scores with the Oracle.
  PYTHONPATH=. python3 scripts/build_reranker_dataset.py --samples 20000 --out data/processed/reranker_dataset.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from utils.condition_vector import get_target_condition
from utils.chemistry import validate_smiles
from models.integrated.pipeline import SafeMolGenDrugOracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator-path", type=str, default="checkpoints/generator/best")
    p.add_argument("--samples", type=int, default=20000, help="Target number of (condition, smiles, score) rows")
    p.add_argument("--batch", type=int, default=200, help="SMILES generated per condition batch")
    p.add_argument("--out", type=str, default="data/processed/reranker_dataset.jsonl")
    p.add_argument("--phases", type=str, default="0.4,0.5,0.55,0.6", help="Comma-separated target phases to sample")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    gen_path = root / args.generator_path if not Path(args.generator_path).is_absolute() else Path(args.generator_path)
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    oracle_path = root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = root / "config" / "endpoints.yaml"
    if not gen_path.exists() or not oracle_path.exists() or not admet_path.exists():
        print("Generator/Oracle/ADMET checkpoints not found.", file=sys.stderr)
        sys.exit(1)

    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    input_dim = get_admet_node_feature_dim(str(admet_path))

    pipeline = SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(gen_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=input_dim,
        device="cpu",
    )
    phases = [float(x.strip()) for x in args.phases.split(",")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while count < args.samples:
            for phase in phases:
                condition = get_target_condition(device="cpu", phase=phase)
                candidates = pipeline.generate_candidates(
                    n=args.batch,
                    temperature=0.8,
                    top_k=40,
                    condition=condition,
                )
                seen = set()
                for smi in candidates:
                    if not validate_smiles(smi) or smi in seen:
                        continue
                    seen.add(smi)
                    pred = pipeline.evaluate_molecule(smi)
                    if pred is None:
                        continue
                    cond_list = condition.flatten().tolist()
                    f.write(json.dumps({
                        "condition": cond_list,
                        "smiles": smi,
                        "oracle_score": pred.overall_prob,
                    }) + "\n")
                    count += 1
                    if count >= args.samples:
                        break
                if count >= args.samples:
                    break
            if count % 5000 == 0 and count > 0:
                print(f"Wrote {count} samples ...", flush=True)
    print(f"Wrote {count} samples to {out_path}. Run scripts/train_reranker.py next.")


if __name__ == "__main__":
    main()
