# Production Training: Best Model Possible

This guide describes how to train SafeMolGen for **production use** (maximum validity, Oracle conditioning, and pipeline success rate), not just a quick demo.

## Quick start

```bash
# One command: pretrain + RL with production defaults
./scripts/run_production_train.sh
```

Or run stages manually:

```bash
# 1. Pretrain (30 epochs, up to 200k SMILES, batch 128)
python scripts/train_generator.py --stage pretrain --production

# 2. RL (15 epochs, batch 16)
python scripts/train_generator.py --stage rl --production --resume checkpoints/generator
```

## What “production” does

| Setting | Demo default | Production |
|--------|----------------|------------|
| Pretrain epochs | 30 | 30 |
| Pretrain data limit | 100k | **200k** |
| Pretrain batch size | 64 | **128** |
| Pretrain checkpoint interval | 10 | **5** |
| RL epochs | 5 | **15** |
| RL batch size | 8 | **16** |
| **Best checkpointing** | last epoch only | **best by validity (pretrain), best by reward (RL)** |

The final checkpoint (`checkpoints/generator` and `checkpoints/generator_rl`) is the **best** model, not the last epoch.

## Data

- **Recommended:** ChEMBL SMILES for scale and diversity.
  ```bash
  python scripts/download_chembl_smiles.py
  ```
  This creates `data/processed/generator/smiles.tsv`. Production pretrain uses up to 200k rows.

- **Fallback:** If that file is missing, the trainer aggregates SMILES from `data/admet_group/*/train_val.csv`. Run `scripts/download_data.py` first so ADMET data exists.

## Optional: larger model

For more capacity (slower, more GPU memory):

```bash
python scripts/train_generator.py --stage pretrain --production \
  --d-model 384 --num-layers 8
```

Then run RL as usual; it will load the larger checkpoint.

## Pipeline and app

- **Pipeline:** By default it loads the generator from `checkpoints/generator` (pretrain best). For the full stack (pretrain + RL), point it at `checkpoints/generator_rl` (see your pipeline config or `design_molecule()` load path).
- **App:** In the sidebar, enable **"Use Oracle-fine-tuned generator"** so the app loads `checkpoints/generator_rl` (the production model after RL).

## Reproducibility

- Seed is fixed with `--seed 42` (default). For identical data order and training, use the same seed and data.
- Best checkpointing is deterministic: the run with the highest validity (pretrain) or reward (RL) is saved.

## Reference

- Config summary: `config/generator_production.yaml`
- Training script: `scripts/train_generator.py` (see `--production`, `--d-model`, `--num-layers`)
