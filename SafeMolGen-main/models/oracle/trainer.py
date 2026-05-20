"""Training loop for DrugOracle."""

from typing import Dict, List

import torch
from torch import nn

from models.oracle.phase_predictors import CascadedPhasePredictors


class OracleTrainer:
    def __init__(
        self,
        model: CascadedPhasePredictors,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            x = batch["x"].to(self.device)
            y1 = batch["phase1"].to(self.device)
            y2 = batch["phase2"].to(self.device)
            y3 = batch["phase3"].to(self.device)

            self.optimizer.zero_grad()
            p1, p2, p3 = self.model(x)
            loss = self.loss_fn(p1, y1) + self.loss_fn(p2, y2) + self.loss_fn(p3, y3)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            count += 1
        return total_loss / max(count, 1)
