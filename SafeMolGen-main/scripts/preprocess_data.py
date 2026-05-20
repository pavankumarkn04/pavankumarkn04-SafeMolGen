"""Preprocess datasets into graph tensors (Phase 1)."""

from pathlib import Path

import pandas as pd
import torch
import yaml

from utils.chemistry import MoleculeProcessor


def _load_endpoint_names(config_path):
    """Load endpoint names from endpoints.yaml (no tdc dependency)."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [item["name"] for item in cfg.get("endpoints", []) if item.get("enabled", True)]


def _build_graphs(df: pd.DataFrame, processor: MoleculeProcessor):
    graphs = []
    for _, row in df.iterrows():
        graph = processor.smiles_to_graph(row["smiles"])
        if graph is None:
            continue
        graph.y = torch.tensor([row["y"]], dtype=torch.float)
        graphs.append(graph)
    return graphs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    endpoints_path = project_root / "config" / "endpoints.yaml"
    endpoint_names = _load_endpoint_names(endpoints_path)
    processor = MoleculeProcessor()

    for ep_name in endpoint_names:
        base_dir = project_root / "data" / "processed" / "admet" / ep_name
        for split in ["train", "val", "test"]:
            csv_path = base_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"Missing {csv_path}, run download_data.py first.")
                continue
            df = pd.read_csv(csv_path)
            graphs = _build_graphs(df, processor)
            out_path = base_dir / f"{split}.pt"
            torch.save(graphs, out_path)
            print(f"Saved {split} graphs for {ep_name}: {len(graphs)}")


if __name__ == "__main__":
    main()
