#!/usr/bin/env python3
"""Train generator by imitation on (condition, best_smiles) from pipeline (plan 3.2).

  PYTHONPATH=. python3 scripts/train_imitation.py --data data/processed/generator/imitation_pairs.jsonl --resume checkpoints/generator/best --out checkpoints/generator_imitation
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import TransformerDecoderModel
from models.generator.trainer import PretrainConfig, train_pretrain
from models.generator.cond_dataset import ImitationDataset
from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/generator/imitation_pairs.jsonl")
    p.add_argument("--resume", type=str, default="checkpoints/generator/best")
    p.add_argument("--out", type=str, default="checkpoints/generator_imitation")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=50000)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    resume_path = root / args.resume if not Path(args.resume).is_absolute() else Path(args.resume)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not data_path.exists():
        print(f"Data not found: {data_path}. Run scripts/collect_imitation_data.py first.", file=sys.stderr)
        sys.exit(1)
    if not (resume_path / "model.pt").exists():
        print(f"Checkpoint not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    pairs: list = []
    with open(data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cond = rec.get("condition")
            smi = rec.get("smiles")
            if not cond or not smi or not validate_smiles(smi):
                continue
            pairs.append((cond, smi))
    if not pairs:
        print("No valid pairs.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(pairs)} (condition, smiles) pairs.")

    gen = SafeMolGen.from_pretrained(str(resume_path), device="cpu")
    tokenizer = gen.tokenizer
    model = gen.model
    smiles_list = [s for _, s in pairs]
    tokenizer.fit(smiles_list)
    dataset = ImitationDataset(pairs, tokenizer)
    config = PretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device="cpu",
        grad_clip=1.0,
        use_cosine_lr=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(resume_path / "model.pt", map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    save_config = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "num_layers": cfg.get("num_layers", 6),
        "dim_feedforward": cfg.get("dim_feedforward", 512),
        "dropout": cfg.get("dropout", 0.1),
        "cond_dim": cfg.get("cond_dim", 0),
    }
    train_pretrain(model, tokenizer, smiles_list, config, dataset=dataset)
    gen = SafeMolGen(tokenizer, model)
    gen.save(str(out_dir), config=save_config)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
