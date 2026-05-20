"""Clinical trial data utilities."""

from pathlib import Path

import pandas as pd


def load_clinical_dataset(path: Path) -> pd.DataFrame:
    """Load clinical dataset with SMILES and phase labels."""
    df = pd.read_csv(path)
    required = {"smiles", "phase1", "phase2", "phase3"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    return df
