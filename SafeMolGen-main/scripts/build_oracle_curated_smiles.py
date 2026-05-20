"""Build Oracle-curated SMILES for target-condition pretraining (Option B).

Scores SMILES from ChEMBL or ADMET aggregate with the DrugOracle, keeps only
high-overall_prob molecules (top fraction or above threshold), and writes a TSV
for use with train_generator.py --use-target-condition.

  PYTHONPATH=. python scripts/build_oracle_curated_smiles.py
  PYTHONPATH=. python scripts/build_oracle_curated_smiles.py --top-pct 20 --out data/processed/generator/smiles_oracle_curated.tsv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from utils.data_utils import read_endpoints_config, load_and_prepare_smiles, aggregate_admet_smiles
from utils.checkpoint_utils import get_admet_node_feature_dim
from utils.chemistry import validate_smiles


def main():
    parser = argparse.ArgumentParser(description="Build Oracle-curated SMILES for target-condition pretrain")
    parser.add_argument("--data", type=str, default="data/processed/generator/smiles.tsv", help="Input SMILES TSV or CSV")
    parser.add_argument("--out", type=str, default="data/processed/generator/smiles_oracle_curated.tsv", help="Output TSV path")
    parser.add_argument("--limit", type=int, default=200_000, help="Max SMILES to score from source")
    parser.add_argument("--top-pct", type=float, default=20.0, help="Keep molecules in top this percent by overall_prob (e.g. 20 = top 20%%)")
    parser.add_argument("--min-overall", type=float, default=None, help="Alternatively: keep molecules with overall_prob >= this (overrides top-pct if set)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    out_path = project_root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = project_root / "config" / "endpoints.yaml"
    if not oracle_path.exists() or not admet_path.exists() or not endpoints_path.exists():
        print("Oracle or ADMET checkpoints not found. Train Oracle first.", file=sys.stderr)
        sys.exit(1)

    from models.oracle.drug_oracle import DrugOracle

    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    input_dim = get_admet_node_feature_dim(str(admet_path))

    print("Loading Oracle...")
    oracle = DrugOracle.from_pretrained(
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        input_dim=input_dim,
        device=args.device,
    )

    if data_path.exists():
        print(f"Loading SMILES from {data_path} (limit={args.limit})...")
        smiles_list = load_and_prepare_smiles(data_path, limit=args.limit, canonicalize=True, write_cleaned_path=None)
    else:
        admet_base = project_root / "data" / "admet_group"
        if not admet_base.exists():
            print(f"Data not found: {data_path} and no {admet_base}", file=sys.stderr)
            sys.exit(1)
        print(f"Aggregating SMILES from {admet_base} (limit={args.limit})...")
        smiles_list = aggregate_admet_smiles(admet_base, limit=args.limit)

    if not smiles_list:
        print("No valid SMILES to score.", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(smiles_list)} SMILES with Oracle...")
    scored = []
    for i, smi in enumerate(smiles_list):
        if (i + 1) % 5000 == 0:
            print(f"  Scored {i + 1}/{len(smiles_list)}...")
        if not validate_smiles(smi):
            continue
        pred = oracle.predict(smi)
        if pred is not None:
            scored.append((smi, pred.overall_prob))

    if not scored:
        print("No scored SMILES.", file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda x: x[1], reverse=True)
    n = len(scored)

    if args.min_overall is not None:
        curated = [(s, p) for s, p in scored if p >= args.min_overall]
        print(f"Filter: overall_prob >= {args.min_overall} -> {len(curated)} molecules")
    else:
        k = max(1, int(n * args.top_pct / 100.0))
        curated = scored[:k]
        print(f"Filter: top {args.top_pct}% -> {len(curated)} molecules (from {n})")

    if not curated:
        print("No molecules passed the filter.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(curated, columns=["smiles", "overall_prob"])
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(curated)} SMILES to {out_path}")
    print(f"  overall_prob range: {curated[-1][1]:.6f} -- {curated[0][1]:.6f}")


if __name__ == "__main__":
    main()
