"""Phase I/II/III predictors for DrugOracle."""

from typing import Tuple

import torch
from torch import nn


class PhasePredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CascadedPhasePredictors(nn.Module):
    """Cascaded predictors: Phase I -> Phase II -> Phase III."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.15):
        super().__init__()
        self.phase1 = PhasePredictor(in_dim, hidden_dim, dropout)
        self.phase2 = PhasePredictor(in_dim + 1, hidden_dim, dropout)
        self.phase3 = PhasePredictor(in_dim + 2, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1 = self.phase1(x)
        p1_sig = torch.sigmoid(p1).unsqueeze(-1)
        p2 = self.phase2(torch.cat([x, p1_sig], dim=-1))
        p2_sig = torch.sigmoid(p2).unsqueeze(-1)
        p3 = self.phase3(torch.cat([x, p1_sig, p2_sig], dim=-1))
        return p1, p2, p3

