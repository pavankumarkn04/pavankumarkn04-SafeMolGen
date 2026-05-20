"""Train two-stage reranker: (condition, SMILES) -> oracle score."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.generator.safemolgen import SafeMolGen
from models.generator.transformer import COND_DIM
from models.reranker.model import RerankerModel
from models.reranker.dataset import RerankerDataset
from utils.chemistry import validate_smiles


def _collate(batch):
    conds, ids_list, scores = zip(*batch)
    conds = torch.stack(conds)
    ids_batch = torch.stack(ids_list)
    scores = torch.tensor(scores, dtype=torch.float32)
    return conds, ids_batch, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/reranker_dataset.jsonl")
    p.add_argument("--generator", type=str, default="checkpoints/generator/best")
    p.add_argument("--out", type=str, default="checkpoints/reranker")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=100000)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    gen_path = root / args.generator if not Path(args.generator).is_absolute() else Path(args.generator)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not data_path.exists():
        print(f"Data not found: {data_path}. Build reranker dataset first.", file=sys.stderr)
        sys.exit(1)
    if not (gen_path / "model.pt").exists():
        print(f"Generator checkpoint not found: {gen_path}", file=sys.stderr)
        sys.exit(1)

    pairs = []
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
            score = rec.get("oracle_score", 0.0)
            if not cond or not smi or not validate_smiles(smi):
                continue
            pairs.append((cond, smi, float(score)))
    if not pairs:
        print("No valid pairs.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(pairs)} (condition, smiles, score) pairs.")

    gen = SafeMolGen.from_pretrained(str(gen_path), device="cpu")
    tokenizer = gen.tokenizer
    vocab_size = tokenizer.vocab_size
    max_len = tokenizer.max_length

    dataset = RerankerDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    model = RerankerModel(
        cond_dim=COND_DIM, vocab_size=vocab_size, max_len=max_len,
        smiles_emb_dim=64, hidden_dim=64, dropout=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for cond, ids, target in loader:
            pred = model(cond, ids)
            loss = mse(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {avg:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config = {
        "cond_dim": COND_DIM, "vocab_size": vocab_size, "max_len": max_len,
        "smiles_emb_dim": 64, "hidden_dim": 64, "dropout": 0.1,
    }
    torch.save({"model": model.state_dict(), "config": save_config}, out_dir / "reranker.pt")
    print(f"Saved to {out_dir / 'reranker.pt'}")


if __name__ == "__main__":
    main()
