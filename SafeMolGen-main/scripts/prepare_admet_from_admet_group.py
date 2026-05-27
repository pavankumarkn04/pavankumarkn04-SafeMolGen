"""Prepare data/processed/admet from existing data/admet_group (Option A2).

Maps endpoint names from config to admet_group folder names (tdc_name lowercased),
splits train_val into train/val, and writes CSVs so preprocess_data.py can run.
Run from project root with PYTHONPATH=. python scripts/prepare_admet_from_admet_group.py
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def _load_endpoints(config_path):
    """Load endpoint name and tdc_name from endpoints.yaml (no tdc dependency)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = []
    for item in cfg.get("endpoints", []):
        if not item.get("enabled", True):
            continue
        out.append({"name": item["name"], "tdc_name": item["tdc_name"], "task_type": item.get("task_type", "regression")})
    return out


def _get_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Drug", "SMILES", "smiles"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find SMILES column in dataset.")


def _admet_group_folder_name(tdc_name: str) -> str:
    """TDC benchmark name to admet_group folder name (e.g. CYP2C9_Substrate_CarbonMangels -> cyp2c9_substrate_carbonmangels)."""
    return tdc_name.lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    admet_base = project_root / "data" / "admet_group"
    processed_base = project_root / "data" / "processed" / "admet"
    endpoints_path = project_root / "config" / "endpoints.yaml"

    if not admet_base.exists():
        raise FileNotFoundError(f"data/admet_group not found at {admet_base}")

    endpoints = _load_endpoints(endpoints_path)
    seed = 42

    for endpoint in endpoints:
        folder_name = _admet_group_folder_name(endpoint["tdc_name"])
        src_dir = admet_base / folder_name
        if not src_dir.exists():
            print(f"Skip {endpoint['name']}: folder {folder_name} not found at {src_dir}")
            continue

        train_val_path = src_dir / "train_val.csv"
        test_path = src_dir / "test.csv"
        if not train_val_path.exists() or not test_path.exists():
            print(f"Skip {endpoint['name']}: missing train_val.csv or test.csv in {src_dir}")
            continue

        train_val = pd.read_csv(train_val_path)
        test_df = pd.read_csv(test_path)
        smiles_col = _get_smiles_col(train_val)
        train_val = train_val.rename(columns={smiles_col: "smiles", "Y": "y"})[["smiles", "y"]]
        test_smiles_col = _get_smiles_col(test_df)
        test_df = test_df.rename(columns={test_smiles_col: "smiles", "Y": "y"})[["smiles", "y"]]

        stratify = None
        if endpoint.get("task_type") == "classification" and train_val["y"].nunique() > 1:
            stratify = train_val["y"]
        train_df, val_df = train_test_split(
            train_val, test_size=0.1, random_state=seed, shuffle=True, stratify=stratify
        )

        out_dir = processed_base / endpoint["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / "train.csv", index=False)
        val_df.to_csv(out_dir / "val.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)
        print(f"Prepared {endpoint['name']} from {folder_name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
