"""RL fine-tuning for SafeMolGen (REINFORCE and PPO)."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm import tqdm

from models.generator.rewards import (
    _alert_penalty,
    _oracle_scalar,
    _scalar_from_prediction,
    compute_rewards,
    compute_rewards_per_sample,
)


@dataclass
class RLConfig:
    epochs: int = 5
    batch_size: int = 8
    accumulation_steps: int = 1
    lr: float = 5e-5
    device: str = "cpu"
    temperature: float = 0.7
    top_k: int = 20
    max_length: Optional[int] = 64
    w_validity: float = 0.75
    w_qed: float = 0.2
    w_oracle: float = 0.1
    w_diversity: float = 0.05
    use_baseline: bool = True
    validity_gated_oracle: bool = True
    phase_weights: Optional[Tuple[float, float, float]] = None
    batch_normalize_oracle: bool = False
    use_ppo: bool = False
    ppo_eps: float = 0.2
    ppo_epochs: int = 3
    use_value_baseline: bool = False
    w_alert: float = 0.0


def _sample_with_logprobs(
    model: nn.Module,
    tokenizer,
    n: int,
    device: str,
    temperature: float,
    top_k: int,
    max_length: Optional[int],
    condition: Optional[torch.Tensor] = None,
) -> Tuple[List[str], torch.Tensor]:
    model.eval()
    max_length = max_length or tokenizer.max_length
    bos_id = tokenizer.vocab[tokenizer.BOS_TOKEN]
    eos_id = tokenizer.vocab[tokenizer.EOS_TOKEN]
    pad_id = tokenizer.vocab[tokenizer.PAD_TOKEN]
    unk_id = tokenizer.vocab[tokenizer.UNK_TOKEN]
    cond_dim = getattr(model, "cond_dim", 0)
    if cond_dim > 0 and condition is None:
        condition = torch.zeros(1, cond_dim, device=device, dtype=torch.float32)

    smiles_list: List[str] = []
    logprobs: List[torch.Tensor] = []
    for _ in range(n):
        ids = [bos_id]
        logprob = torch.tensor(0.0, device=device)
        for _ in range(max_length - 1):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids, condition=condition)[:, -1, :] if cond_dim > 0 else model(input_ids)[:, -1, :]
            logits[:, [bos_id, pad_id, unk_id]] = float("-inf")
            if top_k and top_k > 0:
                top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                min_top = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top, torch.full_like(logits, float("-inf")), logits
                )
            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
                step_logprob = torch.log_softmax(logits, dim=-1)[0, next_id]
            else:
                logits = logits / max(temperature, 1e-6)
                log_probs = torch.log_softmax(logits, dim=-1)
                next_id = int(torch.multinomial(torch.exp(log_probs), num_samples=1).item())
                step_logprob = log_probs[0, next_id]
            ids.append(next_id)
            logprob = logprob + step_logprob
            if next_id == eos_id:
                break
        if len(ids) < max_length:
            ids += [pad_id] * (max_length - len(ids))
        smiles_list.append(tokenizer.decode(ids))
        logprobs.append(logprob)
    return smiles_list, torch.stack(logprobs)


def _logprob_sequences(
    model: nn.Module,
    tokenizer,
    smiles_list: List[str],
    device: str,
    condition: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Compute log P(smiles) under current policy (teacher-forcing). Returns shape (n,)."""
    model.eval()
    max_length = max_length or tokenizer.max_length
    cond_dim = getattr(model, "cond_dim", 0)
    if cond_dim > 0 and condition is None:
        condition = torch.zeros(1, cond_dim, device=device, dtype=torch.float32)
    logprobs_list: List[torch.Tensor] = []
    for smi in smiles_list:
        try:
            ids = tokenizer.encode(smi)
        except Exception:
            logprobs_list.append(torch.tensor(0.0, device=device))
            continue
        if len(ids) < 2:
            logprobs_list.append(torch.tensor(0.0, device=device))
            continue
        ids = ids[: max_length + 1]
        logprob = torch.tensor(0.0, device=device)
        for t in range(1, len(ids)):
            input_ids = torch.tensor([ids[:t]], dtype=torch.long, device=device)
            logits = model(input_ids, condition=condition)[:, -1, :] if cond_dim > 0 else model(input_ids)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            next_id = ids[t]
            logprob = logprob + log_probs[0, next_id]
        logprobs_list.append(logprob)
    return torch.stack(logprobs_list)


def _make_value_net(cond_dim: int, device: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(cond_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)


def train_rl(
    model: nn.Module,
    tokenizer,
    config: RLConfig,
    oracle_score_fn: Optional[Callable[[str], float]] = None,
    oracle_prediction_fn: Optional[Callable[[str], Any]] = None,
    target_condition: Optional[torch.Tensor] = None,
    on_epoch_end: Optional[Callable[[int, float, float], None]] = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(config.device)
    model.train()
    baseline = 0.0
    baseline_momentum = 0.9
    cond_dim = getattr(model, "cond_dim", 0)
    if cond_dim > 0 and target_condition is not None and target_condition.device != config.device:
        target_condition = target_condition.to(config.device)
    value_net = None
    value_optimizer = None
    if config.use_value_baseline and cond_dim > 0 and target_condition is not None:
        value_net = _make_value_net(cond_dim, config.device)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.lr)

    with tqdm(total=config.epochs, desc="RL Fine-tuning") as pbar:
        for epoch in range(1, config.epochs + 1):
            all_smiles: List[str] = []
            all_logprobs: List[torch.Tensor] = []
            all_rewards: List[float] = []
            optimizer.zero_grad()
            for _ in range(config.accumulation_steps):
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
            oracle_override = None
            if config.batch_normalize_oracle and (oracle_score_fn is not None or oracle_prediction_fn is not None):
                if oracle_prediction_fn is not None and config.w_alert != 0:
                    raw_oracle = []
                    for s in smiles_batch:
                        pred = oracle_prediction_fn(s)
                        raw_oracle.append(
                            _scalar_from_prediction(pred, config.phase_weights)
                            - config.w_alert * _alert_penalty(pred)
                        )
                else:
                    raw_oracle = [
                        _oracle_scalar(oracle_score_fn, s, config.phase_weights)
                        for s in smiles_batch
                    ]
                t = torch.tensor(raw_oracle, dtype=torch.float32)
                mean, std = t.mean().item(), t.std().item()
                if std < 1e-8:
                    std = 1.0
                normalized = ((t - mean) / std).clamp(-2.0, 2.0)
                oracle_override = normalized.tolist()
            rewards_per_sample = compute_rewards_per_sample(
                smiles_batch,
                oracle_score_fn=oracle_score_fn if (oracle_prediction_fn is None or config.w_alert == 0) else None,
                w_validity=config.w_validity,
                w_qed=config.w_qed,
                w_oracle=config.w_oracle,
                w_diversity=config.w_diversity,
                validity_gated_oracle=config.validity_gated_oracle,
                phase_weights=config.phase_weights,
                oracle_scores_override=oracle_override,
                oracle_prediction_fn=oracle_prediction_fn if config.w_alert != 0 else None,
                w_alert=config.w_alert,
            )
            all_smiles.extend(smiles_batch)
            all_logprobs.append(logprobs)
            all_rewards.extend(rewards_per_sample)
            logprobs_cat = torch.cat(all_logprobs, dim=0)
            reward_tensor = torch.tensor(all_rewards, device=config.device)
            if value_net is not None and target_condition is not None:
                cond_expand = target_condition.expand(logprobs_cat.size(0), -1)
                value_pred = value_net(cond_expand).squeeze(-1)
                advantage = reward_tensor - value_pred.detach()
                value_loss = (reward_tensor - value_pred).pow(2).mean()
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
            elif config.use_baseline:
                batch_mean = reward_tensor.mean().item()
                baseline = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean
                advantage = reward_tensor - baseline
            else:
                advantage = reward_tensor
            advantage = advantage.to(config.device)
            old_logprobs = logprobs_cat.detach()
            if config.use_ppo and config.ppo_epochs > 0:
                for _ in range(config.ppo_epochs):
                    optimizer.zero_grad()
                    new_logprobs = _logprob_sequences(
                        model,
                        tokenizer,
                        all_smiles,
                        config.device,
                        condition=target_condition if cond_dim > 0 else None,
                        max_length=config.max_length,
                    )
                    ratio = torch.exp(new_logprobs - old_logprobs)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(
                        ratio, 1.0 - config.ppo_eps, 1.0 + config.ppo_eps
                    ) * advantage
                    loss = -torch.min(surr1, surr2).mean()
                    loss.backward()
                    optimizer.step()
            else:
                loss = -(advantage * logprobs_cat).mean()
                loss.backward()
                optimizer.step()
            metrics = compute_rewards(
                all_smiles,
                oracle_score_fn=oracle_score_fn,
                w_validity=config.w_validity,
                w_qed=config.w_qed,
                w_oracle=config.w_oracle,
                w_diversity=config.w_diversity,
                validity_gated_oracle=config.validity_gated_oracle,
                phase_weights=config.phase_weights,
            )
            batch_reward = metrics["total"]
            validity = metrics["validity"]
            pbar.set_postfix(reward=f"{batch_reward:.4f}", validity=f"{validity:.2%}")
            pbar.update(1)
            print(f"RL Epoch {epoch} | Reward: {batch_reward:.4f} | Validity: {validity:.1%}")
            if on_epoch_end is not None:
                on_epoch_end(epoch, batch_reward, validity)
