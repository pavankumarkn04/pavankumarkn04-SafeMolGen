"""SafeMolGen generator model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import CausalTransformerGenerator, COND_DIM


@dataclass
class GenerationConfig:
    max_length: int = 128
    temperature: float = 0.75
    top_k: int = 40
    max_attempts_per_sample: int = 20


class SafeMolGen:
    def __init__(self, tokenizer: SMILESTokenizer, model: CausalTransformerGenerator):
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        max_len_override: Optional[int] = None,
    ) -> "SafeMolGen":
        base = Path(path)
        tokenizer = SMILESTokenizer.load(base / "tokenizer.json")
        state = torch.load(base / "model.pt", map_location=device, weights_only=False)
        checkpoint_max_len = state["model"]["positional.pe"].shape[1]
        max_len = max_len_override or checkpoint_max_len
        cfg = state.get("config", {})
        cond_dim = cfg.get("cond_dim", 0)
        model = CausalTransformerGenerator(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("num_layers", 6),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
            max_len=max_len,
            cond_dim=cond_dim,
        )
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        model.eval()
        return cls(tokenizer, model)

    def save(self, path: str, config: dict) -> None:
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(base / "tokenizer.json")
        torch.save({"model": self.model.state_dict(), "config": config}, base / "model.pt")

    def generate(
        self,
        n: int = 10,
        temperature: float = 0.75,
        max_length: Optional[int] = None,
        device: str = "cpu",
        top_k: int = 40,
        disallow_special: bool = True,
        condition: Optional[torch.Tensor] = None,
    ) -> List[str]:
        self.model.eval()
        max_length = max_length or self.tokenizer.max_length
        bos_id = self.tokenizer.vocab[self.tokenizer.BOS_TOKEN]
        eos_id = self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]
        pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
        unk_id = self.tokenizer.vocab[self.tokenizer.UNK_TOKEN]
        if condition is not None and condition.device != device:
            condition = condition.to(device)

        generated = []
        for _ in range(n):
            ids = [bos_id]
            for _ in range(max_length - 1):
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                logits = self.model(input_ids, condition=condition)[:, -1, :]
                if disallow_special:
                    logits[:, [bos_id, pad_id, unk_id]] = float("-inf")
                if top_k and top_k > 0:
                    top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                    min_top = top_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top, torch.full_like(logits, float("-inf")), logits)
                if temperature <= 0:
                    next_id = int(torch.argmax(logits, dim=-1).item())
                else:
                    logits = logits / max(temperature, 1e-6)
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                ids.append(next_id)
                if next_id == eos_id:
                    break
            if len(ids) < max_length:
                ids += [pad_id] * (max_length - len(ids))
            generated.append(self.tokenizer.decode(ids))
        return generated

    def generate_valid(
        self,
        n: int = 10,
        temperature: float = 0.75,
        max_length: Optional[int] = None,
        device: str = "cpu",
        top_k: int = 40,
        max_attempts_per_sample: int = 20,
    ) -> List[str]:
        from utils.chemistry import validate_smiles

        valid = []
        attempts = 0
        max_attempts = max(n * max_attempts_per_sample, n)
        while len(valid) < n and attempts < max_attempts:
            sample = self.generate(
                n=1,
                temperature=temperature,
                max_length=max_length,
                device=device,
                top_k=top_k,
                disallow_special=True,
            )[0]
            attempts += 1
            if validate_smiles(sample):
                valid.append(sample)
        return valid

    def generate_with_constraints(self, props: dict, n: int = 10) -> List[str]:
        return self.generate(n=n)

    def generate_iterative(self, oracle_score_fn, n_iter: int = 5, n: int = 50) -> List[str]:
        best = []
        best_score = -1.0
        for _ in range(n_iter):
            candidates = self.generate(n=n)
            scores = [oracle_score_fn(s) for s in candidates]
            if scores:
                max_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                if scores[max_idx] > best_score:
                    best_score = scores[max_idx]
                    best = [candidates[max_idx]]
        return best
