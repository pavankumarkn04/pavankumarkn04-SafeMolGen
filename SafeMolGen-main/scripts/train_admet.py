"""Train ADMET predictor (Phase 1)."""

from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from models.admet.multi_task_predictor import MultiTaskADMETPredictor
from models.admet.trainer import ADMETTrainer
from utils.data_utils import read_endpoints_config
from utils.logging_config import setup_logging


def _load_graphs(path: Path):
    return torch.load(path, weights_only=False)


def main() -> None:
    setup_logging()
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
    epochs = cfg["training"]["epochs"]

    loaders_train = {}
    loaders_val = {}

    for ep in endpoints:
        base_dir = project_root / "data" / "processed" / "admet" / ep.name
        train_graphs = _load_graphs(base_dir / "train.pt")
        val_graphs = _load_graphs(base_dir / "val.pt")
        loaders_train[ep.name] = DataLoader(
            train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        loaders_val[ep.name] = DataLoader(
            val_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    sample_graph = next(iter(loaders_train[endpoint_names[0]]))
    num_node_features = sample_graph.x.size(-1)

    model = MultiTaskADMETPredictor(
        num_node_features=num_node_features,
        hidden_dim=cfg["model"]["gnn_hidden_dim"],
        num_layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
        endpoint_names=endpoint_names,
    )

    endpoint_weights = cfg.get("admet_endpoint_weights") or {}
    trainer = ADMETTrainer(
        model=model,
        endpoint_configs=endpoint_dicts,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        endpoint_weights=endpoint_weights,
    )

    best_score = -1e9
    checkpoints_dir = project_root / "checkpoints" / "admet"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )

    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(loaders_train)
        metrics = trainer.evaluate(loaders_val)

        scores = []
        for ep in endpoint_dicts:
            name = ep["name"]
            metric_key = ep.get("metric", "roc_auc")
            value = metrics[name].get(metric_key, float("nan"))
            if metric_key in {"rmse", "mae"}:
                value = -value
            scores.append(value)
        avg_score = float(torch.tensor(scores).nanmean().item())
        scheduler.step(avg_score)

        print(f"Epoch {epoch} | Loss: {loss:.4f} | Score: {avg_score:.4f} | LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

        if avg_score > best_score:
            best_score = avg_score
            torch.save(
                {"model": model.state_dict(), "metrics": metrics},
                checkpoints_dir / "best_model.pt",
            )


if __name__ == "__main__":
    main()
