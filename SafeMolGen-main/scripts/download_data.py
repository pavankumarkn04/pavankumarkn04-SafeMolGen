"""Download datasets for Phase 1 (TDC ADMET)."""

from pathlib import Path

import pandas as pd
import yaml

from utils.data_utils import TDCDataLoader, read_endpoints_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    endpoints_path = project_root / "config" / "endpoints.yaml"
    with open(endpoints_path, "r", encoding="utf-8") as f:
        endpoints_cfg = yaml.safe_load(f)

    endpoints = read_endpoints_config(endpoints_cfg)
    loader = TDCDataLoader(project_root / "data")

    for endpoint in endpoints:
        train_df, val_df, test_df = loader.fetch_endpoint_splits(endpoint)
        raw_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        loader.save_raw(endpoint, raw_df)
        loader.save_splits(endpoint, train_df, val_df, test_df)
        print(f"Downloaded and split: {endpoint.name} ({endpoint.tdc_name})")


if __name__ == "__main__":
    main()
