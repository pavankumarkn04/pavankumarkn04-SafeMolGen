"""Small reranker: (condition, SMILES) -> scalar oracle score."""

from typing import Optional

import torch
from torch import nn

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import COND_DIM


class RerankerModel(nn.Module):
    """Maps (condition vector, SMILES token ids) to predicted oracle score (0-1)."""

    def __init__(
        self,
        cond_dim: int = COND_DIM,
        vocab_size: int = 64,
        max_len: int = 128,
        smiles_emb_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.max_len = max_len
        self.smiles_emb = nn.Embedding(vocab_size, smiles_emb_dim, padding_idx=0)
        self.smiles_encoder = nn.LSTM(
            smiles_emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
            dropout=0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        condition: torch.Tensor,
        smiles_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.smiles_emb(smiles_ids)
        out, _ = self.smiles_encoder(x)
        pooled = out.mean(dim=1)
        combined = torch.cat([condition, pooled], dim=-1)
        return self.mlp(combined).squeeze(-1)


def load_reranker(
    path: str,
    tokenizer: SMILESTokenizer,
    device: str = "cpu",
) -> RerankerModel:
    """Load reranker from checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    model = RerankerModel(
        cond_dim=cfg.get("cond_dim", COND_DIM),
        vocab_size=cfg.get("vocab_size", len(tokenizer.vocab)),
        max_len=cfg.get("max_len", tokenizer.max_length),
        smiles_emb_dim=cfg.get("smiles_emb_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state.get("model", state), strict=True)
    model.to(device)
    model.eval()
    return model
