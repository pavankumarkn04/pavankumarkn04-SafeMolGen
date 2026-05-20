#!/usr/bin/env python3
"""Best-of-N fine-tune (plan 3.1): sample N, weight log-prob by oracle score, maximize weighted MLE.

  PYTHONPATH=. python3 scripts/train_best_of_n.py --resume checkpoints/generator/best --out checkpoints/generator_best_of_n --epochs 20
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

from models.generator.safemolgen import SafeMolGen
from models.generator.best_of_n_trainer import BestOfNConfig, train_best_of_n
from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from utils.condition_vector import get_target_condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default="checkpoints/generator/best", help="Generator checkpoint (Option B best or pretrain)")
    p.add_argument("--out", type=str, default="checkpoints/generator_best_of_n", help="Output dir")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-scheme", choices=["softmax", "normalize"], default="softmax")
    p.add_argument("--weight-temperature", type=float, default=0.1)
    p.add_argument("--no-valid-only", action="store_true", help="Weight invalid SMILES by oracle too")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    resume = root / args.resume if not Path(args.resume).is_absolute() else Path(args.resume)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not (resume / "model.pt").exists():
        print(f"Checkpoint not found: {resume}", file=sys.stderr)
        sys.exit(1)

    oracle_path = root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = root / "config" / "endpoints.yaml"
    if not oracle_path.exists() or not admet_path.exists():
        print("Oracle/ADMET checkpoints not found.", file=sys.stderr)
        sys.exit(1)

    from models.oracle.drug_oracle import DrugOracle

    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    input_dim = get_admet_node_feature_dim(str(admet_path))
    oracle = DrugOracle.from_pretrained(
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        input_dim=input_dim,
        device="cpu",
    )

    def oracle_score_fn(smiles: str) -> float:
        pred = oracle.predict(smiles)
        return pred.overall_prob if pred else 0.0

    gen = SafeMolGen.from_pretrained(str(resume), device="cpu")
    model, tokenizer = gen.model, gen.tokenizer
    target_condition = None
    if getattr(model, "cond_dim", 0) > 0:
        target_condition = get_target_condition(device="cpu", phase=0.6)
        print("Using target condition (phase=0.6) for sampling.")

    config = BestOfNConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cpu",
        weight_scheme=args.weight_scheme,
        weight_temperature=args.weight_temperature,
        valid_only=not args.no_valid_only,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(resume / "model.pt", map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    save_config = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "num_layers": cfg.get("num_layers", 6),
        "dim_feedforward": cfg.get("dim_feedforward", 512),
        "dropout": cfg.get("dropout", 0.1),
        "cond_dim": cfg.get("cond_dim", 0),
    }

    def on_epoch_end(epoch: int, loss: float, mean_oracle: float) -> None:
        g = SafeMolGen(tokenizer, model)
        g.save(str(out_dir), config=save_config)

    train_best_of_n(
        model,
        tokenizer,
        config=config,
        oracle_score_fn=oracle_score_fn,
        target_condition=target_condition,
        on_epoch_end=on_epoch_end,
    )
    gen = SafeMolGen.from_pretrained(str(out_dir), device="cpu")
    gen.save(str(out_dir), config=save_config)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
