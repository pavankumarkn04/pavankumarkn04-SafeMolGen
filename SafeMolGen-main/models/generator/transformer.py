"""Causal transformer generator for SMILES generation."""

from typing import Optional

import torch
from torch import nn

from models.generator.positional_encoding import PositionalEncoding

COND_DIM = 25


class CausalTransformerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, d_model)
        else:
            self.cond_proj = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(input_ids)
        x = self.positional(x)
        if self.cond_proj is not None and condition is not None:
            cond_bias = self.cond_proj(condition).unsqueeze(1)
            x = x + cond_bias
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        x = self.transformer(x, mask)
        return self.fc_out(x)


# Backward-compat alias for old imports/checkpoints.
TransformerDecoderModel = CausalTransformerGenerator
