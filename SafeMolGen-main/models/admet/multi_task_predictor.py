"""Multi-task ADMET predictor."""

from typing import Dict, List

import torch
from torch import nn

from models.admet.attention_pooling import AttentionPooling
from models.admet.gnn_encoder import GNNEncoder


class MultiTaskADMETPredictor(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        endpoint_names: List[str],
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(num_node_features, hidden_dim, num_layers, dropout)
        self.pool = AttentionPooling(hidden_dim)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in endpoint_names})

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x = self.encoder(x, edge_index)
        pooled = self.pool(x, batch)
        return {name: head(pooled).squeeze(-1) for name, head in self.heads.items()}

