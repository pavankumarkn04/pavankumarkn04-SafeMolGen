"""Data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.chemistry import validate_smiles


@dataclass
class EndpointConfig:
    name: str
    category: str
    tdc_name: str
    task_type: str
    metric: str
    enabled: bool = True


def read_endpoints_config(config: Dict) -> List[EndpointConfig]:
    endpoints = []
    for item in config.get("endpoints", []):
        if not item.get("enabled", True):
            continue
        endpoints.append(
            EndpointConfig(
                name=item["name"],
                category=item["category"],
                tdc_name=item["tdc_name"],
                task_type=item["task_type"],
                metric=item.get("metric", "rmse"),
                enabled=item.get("enabled", True),
            )
        )
    return endpoints


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_smiles_column(df: pd.DataFrame) -> str:
    for col in ["Drug", "SMILES", "smiles"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find SMILES column in dataset.")


class TDCDataLoader:
    """Download and prepare TDC ADMET datasets."""

    def __init__(self, data_dir: Path, seed: int = 42) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self.group = None

    def _ensure_group(self):
        if self.group is None:
            from tdc.benchmark_group import admet_group
            self.group = admet_group(path=str(self.data_dir))

    def fetch_endpoint_splits(
        self, endpoint: EndpointConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._ensure_group()
        benchmark = self.group.get(endpoint.tdc_name)
        train_val = benchmark["train_val"]
        test_df = benchmark["test"]

        smiles_col = _get_smiles_column(train_val)
        train_val = train_val.rename(columns={smiles_col: "smiles", "Y": "y"})
        test_df = test_df.rename(columns={smiles_col: "smiles", "Y": "y"})

        train_val = train_val[["smiles", "y"]]
        test_df = test_df[["smiles", "y"]]

        stratify: Optional[pd.Series] = None
        if endpoint.task_type == "classification":
            if train_val["y"].nunique() > 1:
                stratify = train_val["y"]

        train_df, val_df = train_test_split(
            train_val,
            test_size=0.1,
            random_state=self.seed,
            shuffle=True,
            stratify=stratify,
        )
        return train_df, val_df, test_df

    def save_raw(self, endpoint: EndpointConfig, df: pd.DataFrame) -> Path:
        raw_dir = self.data_dir / "raw"
        ensure_dir(raw_dir)
        path = raw_dir / f"tdc_{endpoint.name}.csv"
        df.to_csv(path, index=False)
        return path

    def save_splits(
        self,
        endpoint: EndpointConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[Path, Path, Path]:
        processed_dir = self.data_dir / "processed" / "admet" / endpoint.name
        ensure_dir(processed_dir)

        train_path = processed_dir / "train.csv"
        val_path = processed_dir / "val.csv"
        test_path = processed_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        return train_path, val_path, test_path


def _get_smiles_series(df: pd.DataFrame):
    for col in ["canonical_smiles", "smiles", "SMILES", "Drug"]:
        if col in df.columns:
            return df[col].dropna()
    return df.iloc[:, 1].dropna()


def load_and_prepare_smiles(
    path: Path,
    limit: int = 50000,
    canonicalize: bool = True,
    write_cleaned_path: Optional[Path] = None,
) -> List[str]:
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    raw = _get_smiles_series(df).astype(str).tolist()
    raw = [s.strip() for s in raw if s.strip()][: limit * 2]
    valid = [s for s in raw if validate_smiles(s)]
    n_raw, n_valid = len(raw), len(valid)
    if n_raw == 0:
        return []
    pct = 100.0 * n_valid / n_raw
    print(f"Loaded {n_raw} SMILES, {n_valid} valid ({pct:.1f}%)")
    if n_valid == 0:
        return []
    if canonicalize:
        try:
            from rdkit import Chem
            canonical = []
            for s in valid:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    canonical.append(Chem.MolToSmiles(mol))
            valid = canonical
            print(f"Canonicalized: {len(valid)} SMILES")
        except Exception as e:
            print(f"Canonicalize skipped: {e}")
    result = list(dict.fromkeys(valid))[:limit]
    if write_cleaned_path is not None and result:
        write_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": result}).to_csv(
            write_cleaned_path, index=False, header=True
        )
        print(f"Wrote cleaned SMILES to {write_cleaned_path}")
    return result


def aggregate_admet_smiles(admet_base: Path, limit: int = 50000) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for csv_path in sorted(admet_base.glob("*/train_val.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            series = _get_smiles_series(df).astype(str)
            for s in series:
                s = s.strip()
                if not s or s in seen:
                    continue
                if validate_smiles(s):
                    seen.add(s)
                    out.append(s)
                    if len(out) >= limit:
                        break
        except Exception as e:
            print(f"Skip {csv_path}: {e}")
        if len(out) >= limit:
            break
    print(f"Aggregated {len(out)} valid SMILES from ADMET (fallback)")
    return out
