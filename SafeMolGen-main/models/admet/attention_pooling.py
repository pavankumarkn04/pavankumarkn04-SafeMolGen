"""Attention pooling layer."""

import torch
from torch import nn
from torch_geometric.nn import GlobalAttention


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
        self.pool = GlobalAttention(gate_nn)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.pool(x, batch)

