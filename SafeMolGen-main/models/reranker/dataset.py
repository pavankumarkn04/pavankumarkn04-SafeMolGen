"""Dataset for reranker: (condition, smiles, oracle_score)."""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import COND_DIM


class RerankerDataset(Dataset):
    """Yields (condition, smiles_ids, score) for RerankerModel."""

    def __init__(
        self,
        pairs: List[Tuple[List[float], str, float]],
        tokenizer: SMILESTokenizer,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        cond_list, smi, score = self.pairs[idx]
        ids = self.tokenizer.encode(smi)
        cond = torch.tensor(cond_list[:COND_DIM], dtype=torch.float32)
        if cond.numel() < COND_DIM:
            cond = torch.nn.functional.pad(cond, (0, COND_DIM - cond.numel()), value=0.0)
        ids_t = torch.tensor(ids, dtype=torch.long)
        return cond, ids_t, float(score)
