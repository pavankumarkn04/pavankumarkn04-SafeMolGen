"""Evaluate ADMET predictor (Phase 1)."""

from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from models.admet.multi_task_predictor import MultiTaskADMETPredictor
from models.admet.trainer import ADMETTrainer
from utils.data_utils import read_endpoints_config


def _load_graphs(path: Path):
    return torch.load(path, weights_only=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(project_root / "config" / "endpoints.yaml", "r", encoding="utf-8") as f:
        endpoints_cfg = yaml.safe_load(f)

    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_dicts = [e.__dict__ for e in endpoints]

    device_name = cfg["training"]["device"]
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    loaders_test = {}
    for ep in endpoints:
        base_dir = project_root / "data" / "processed" / "admet" / ep.name
        test_graphs = _load_graphs(base_dir / "test.pt")
        loaders_test[ep.name] = DataLoader(
            test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    sample_graph = next(iter(loaders_test[endpoint_names[0]]))
    num_node_features = sample_graph.x.size(-1)

    model = MultiTaskADMETPredictor(
        num_node_features=num_node_features,
        hidden_dim=cfg["model"]["gnn_hidden_dim"],
        num_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        endpoint_names=endpoint_names,
    )
    checkpoint = torch.load(
        project_root / "checkpoints" / "admet" / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])

    trainer = ADMETTrainer(
        model=model,
        endpoint_configs=endpoint_dicts,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    metrics = trainer.evaluate(loaders_test)
    for name, vals in metrics.items():
        print(name, vals)


if __name__ == "__main__":
    main()
