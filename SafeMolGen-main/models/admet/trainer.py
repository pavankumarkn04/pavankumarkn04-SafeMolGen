"""Training logic for ADMET predictor."""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from models.admet.losses import get_loss
from utils.metrics import compute_metrics


class ADMETTrainer:
    def __init__(
        self,
        model: nn.Module,
        endpoint_configs: List[Dict],
        device: torch.device,
        lr: float,
        weight_decay: float,
        endpoint_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model.to(device)
        self.endpoint_configs = endpoint_configs
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fns = {
            ep["name"]: get_loss(ep["task_type"]) for ep in endpoint_configs
        }
        self.endpoint_weights = endpoint_weights or {}

    def train_epoch(self, loaders: Dict[str, torch.utils.data.DataLoader]) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for endpoint in self.endpoint_configs:
            name = endpoint["name"]
            loss_fn = self.loss_fns[name]
            w = self.endpoint_weights.get(name, 1.0)
            loader = loaders[name]
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                preds = outputs[name]
                loss = w * loss_fn(preds, batch.y.view_as(preds))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def evaluate(
        self, loaders: Dict[str, torch.utils.data.DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for endpoint in self.endpoint_configs:
                name = endpoint["name"]
                task_type = endpoint["task_type"]
                y_true = []
                y_pred = []
                for batch in loaders[name]:
                    batch = batch.to(self.device)
                    outputs = self.model(batch.x, batch.edge_index, batch.batch)
                    preds = outputs[name].detach().cpu().numpy()
                    if task_type == "classification":
                        preds = 1 / (1 + np.exp(-preds))
                    y_pred.extend(preds.tolist())
                    y_true.extend(batch.y.cpu().numpy().tolist())
                metrics[name] = compute_metrics(
                    np.array(y_true), np.array(y_pred), task_type
                )
        return metrics
