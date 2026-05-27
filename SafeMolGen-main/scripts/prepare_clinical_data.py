"""Create data/processed/oracle/clinical_trials.csv for Oracle training (Phase 2).

Rule-based version: when ADMET checkpoint exists, compute phase1/phase2/phase3 from
deterministic rules over ADMET predictions so "better ADMET" yields higher phases.
Falls back to random labels only when ADMET is not available. Run from project root with PYTHONPATH=.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


def _get_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Drug", "SMILES", "smiles"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find SMILES column.")


def _rule_based_phases(admet: dict) -> Tuple[float, float, float]:
    """Compute phase1, phase2, phase3 from ADMET dict (deterministic, so better ADMET -> higher phases)."""
    # Safe get with defaults; classification endpoints are 0-1, regression may vary
    def g(k: str, default: float = 0.5) -> float:
        v = admet.get(k, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Phase I: safety & bioavailability (high bioav, low hERG, low AMES -> high phase1)
    bioav = g("bioavailability_ma", 0.5)
    herg = g("herg", 0.5)
    ames = g("ames", 0.5)
    phase1 = (bioav + (1.0 - herg) + (1.0 - ames)) / 3.0
    phase1 = max(0.0, min(1.0, phase1))

    # Phase II: Phase I plus metabolism/clearance (favorable CYP/clearance -> higher phase2)
    cyp_risk = (g("cyp3a4_veith", 0.5) + g("cyp2d6_veith", 0.5)) / 2.0
    phase2 = phase1 * (0.4 + 0.6 * (1.0 - cyp_risk))
    phase2 = max(0.0, min(1.0, phase2))

    # Phase III: Phase II plus DILI/toxicity (low DILI -> higher phase3)
    dili = g("dili", 0.5)
    phase3 = phase2 * (0.5 + 0.5 * (1.0 - dili))
    phase3 = max(0.0, min(1.0, phase3))

    return (phase1, phase2, phase3)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    admet_base = project_root / "data" / "admet_group"
    out_path = project_root / "data" / "processed" / "oracle" / "clinical_trials.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique valid SMILES from admet_group
    seen = set()
    smiles_list = []
    for csv_path in sorted(admet_base.glob("*/train_val.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            col = _get_smiles_col(df)
            for s in df[col].astype(str).dropna():
                s = s.strip()
                if s and s not in seen:
                    try:
                        from rdkit import Chem
                        if Chem.MolFromSmiles(s) is not None:
                            seen.add(s)
                            smiles_list.append(s)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Skip {csv_path}: {e}")
        if len(smiles_list) >= 10000:
            break

    if len(smiles_list) < 100:
        raise ValueError(f"Too few valid SMILES ({len(smiles_list)}). Need at least 100.")

    smiles_list = smiles_list[:5000]
    use_rule_based = False
    admet_ckpt = project_root / "checkpoints" / "admet" / "best_model.pt"
    if admet_ckpt.exists():
        try:
            from models.admet.inference import load_model, predict_smiles
            from utils.data_utils import read_endpoints_config
            from utils.checkpoint_utils import get_admet_node_feature_dim

            endpoints_cfg = yaml.safe_load(
                (project_root / "config" / "endpoints.yaml").read_text(encoding="utf-8")
            )
            endpoints = read_endpoints_config(endpoints_cfg)
            endpoint_names = [e.name for e in endpoints]
            endpoint_task_types = {e.name: e.task_type for e in endpoints}
            num_node_features = get_admet_node_feature_dim(str(admet_ckpt))
            admet_model = load_model(
                checkpoint_path=str(admet_ckpt),
                endpoint_names=endpoint_names,
                num_node_features=num_node_features,
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
                device="cpu",
            )
            rows = []
            for i, smi in enumerate(smiles_list):
                preds = predict_smiles(
                    admet_model, smi, endpoint_task_types, device="cpu"
                )
                if not preds:
                    continue
                p1, p2, p3 = _rule_based_phases(preds)
                rows.append({"smiles": smi, "phase1": p1, "phase2": p2, "phase3": p3})
                if (i + 1) % 500 == 0:
                    print(f"  Rule-based labels: {i + 1}/{len(smiles_list)}")
            if rows:
                df = pd.DataFrame(rows)
                use_rule_based = True
        except Exception as e:
            print(f"Rule-based labels failed ({e}), falling back to random.")
            use_rule_based = False

    if not use_rule_based:
        import random
        random.seed(42)
        rows = []
        for smi in smiles_list:
            phase1 = float(random.random() > 0.3)
            phase2 = float(random.random() > 0.5) if phase1 > 0.5 else 0.0
            phase3 = float(random.random() > 0.6) if phase2 > 0.5 else 0.0
            rows.append({"smiles": smi, "phase1": phase1, "phase2": phase2, "phase3": phase3})
        df = pd.DataFrame(rows)
        print("Using random phase labels (run after training ADMET for rule-based labels).")

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}" + (" (rule-based)" if use_rule_based else " (random)"))


if __name__ == "__main__":
    main()
