"""Pretraining utilities for SafeMolGen."""

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import CausalTransformerGenerator


class SMILESDataset(Dataset):
    def __init__(self, smiles_list: List[str], tokenizer: SMILESTokenizer):
        self.smiles = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.smiles[idx])
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, target_ids


@dataclass
class PretrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cpu"
    grad_clip: float = 1.0
    use_cosine_lr: bool = True
    shuffle_seed: Optional[int] = None


def train_pretrain(
    model: CausalTransformerGenerator,
    tokenizer: SMILESTokenizer,
    smiles_list: List[str],
    config: PretrainConfig,
    on_epoch_end: Optional[Callable[[int, CausalTransformerGenerator, SMILESTokenizer], None]] = None,
    dataset: Optional[Dataset] = None,
) -> None:
    smiles_list = list(smiles_list)
    if config.shuffle_seed is not None:
        rng = random.Random(config.shuffle_seed)
        rng.shuffle(smiles_list)
    if dataset is None:
        dataset = SMILESDataset(smiles_list, tokenizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.use_cosine_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.PAD_TOKEN])

    model.to(config.device)
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        count = 0
        with tqdm(total=len(loader), desc=f"Pretrain Epoch {epoch}/{config.epochs}") as pbar:
            for batch in loader:
                if len(batch) == 3:
                    input_ids, target_ids, condition = batch
                    input_ids = input_ids.to(config.device)
                    target_ids = target_ids.to(config.device)
                    condition = condition.to(config.device) if condition is not None else None
                else:
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(config.device)
                    target_ids = target_ids.to(config.device)
                    condition = None
                    if getattr(model, "cond_dim", 0) > 0:
                        condition = torch.zeros(
                            input_ids.size(0), model.cond_dim, device=config.device, dtype=torch.float32
                        )
                logits = model(input_ids, condition=condition)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                total_loss += float(loss.item())
                count += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
        if scheduler is not None:
            scheduler.step()
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, tokenizer)
