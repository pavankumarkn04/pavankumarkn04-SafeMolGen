"""Single-task ADMET predictor."""

import torch
from torch import nn

from models.admet.attention_pooling import AttentionPooling
from models.admet.gnn_encoder import GNNEncoder


class SingleTaskADMETPredictor(nn.Module):
    def __init__(
        self, num_node_features: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(num_node_features, hidden_dim, num_layers, dropout)
        self.pool = AttentionPooling(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        x = self.encoder(x, edge_index)
        pooled = self.pool(x, batch)
        return self.head(pooled).squeeze(-1)
