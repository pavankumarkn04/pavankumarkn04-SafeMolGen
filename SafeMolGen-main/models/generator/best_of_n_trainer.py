"""Best-of-N fine-tuning (reward-weighted MLE): no policy gradient, stable supervised updates."""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn
from tqdm import tqdm

from models.generator.rl_trainer import _sample_with_logprobs
from utils.chemistry import validate_smiles


@dataclass
class BestOfNConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-5
    device: str = "cpu"
    temperature: float = 0.8
    top_k: int = 40
    max_length: Optional[int] = None
    weight_scheme: str = "softmax"
    weight_temperature: float = 0.1
    valid_only: bool = True


def train_best_of_n(
    model: nn.Module,
    tokenizer,
    config: BestOfNConfig,
    oracle_score_fn: Callable[[str], float],
    target_condition: Optional[torch.Tensor] = None,
    on_epoch_end: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    """Train generator with weighted MLE: sample N, weight log-prob by oracle score."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(config.device)
    model.train()
    cond_dim = getattr(model, "cond_dim", 0)
    if cond_dim > 0 and target_condition is not None and target_condition.device != config.device:
        target_condition = target_condition.to(config.device)

    with tqdm(total=config.epochs, desc="Best-of-N") as pbar:
        for epoch in range(1, config.epochs + 1):
            smiles_batch, logprobs = _sample_with_logprobs(
                model,
                tokenizer,
                n=config.batch_size,
                device=config.device,
                temperature=config.temperature,
                top_k=config.top_k,
                max_length=config.max_length,
                condition=target_condition if cond_dim > 0 else None,
            )
            scores = []
            for s in smiles_batch:
                try:
                    sc = float(oracle_score_fn(s))
                except Exception:
                    sc = 0.0
                scores.append(sc)

            score_tensor = torch.tensor(scores, device=config.device, dtype=torch.float32)
            if config.valid_only:
                valid_mask = torch.tensor(
                    [1.0 if validate_smiles(s) else 0.0 for s in smiles_batch],
                    device=config.device,
                    dtype=torch.float32,
                )
                score_tensor = score_tensor * valid_mask

            if config.weight_scheme == "softmax":
                weights = torch.softmax(score_tensor / max(config.weight_temperature, 1e-6), dim=0)
            else:
                weights = score_tensor / (score_tensor.sum() + 1e-8)

            loss = -(weights * logprobs).sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            mean_score = score_tensor.mean().item()
            n_valid = sum(1 for s in smiles_batch if validate_smiles(s))
            validity = n_valid / max(len(smiles_batch), 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", oracle=f"{mean_score:.4f}", valid=f"{validity:.2%}")
            pbar.update(1)
            if on_epoch_end is not None:
                on_epoch_end(epoch, float(loss.item()), mean_score)
            print(f"Best-of-N Epoch {epoch} | Loss: {loss.item():.4f} | Mean oracle: {mean_score:.4f} | Validity: {validity:.1%}")
