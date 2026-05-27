#!/usr/bin/env bash
# Production training: pretrain then RL with best checkpointing.
# Ensures generator data exists, then runs 30ep pretrain (200k SMILES, batch 128)
# and 15ep RL (batch 16). Output: checkpoints/generator, checkpoints/generator_rl.

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== Production training: SafeMolGen ==="

# 1. Ensure some SMILES data exists (ChEMBL or ADMET fallback)
DATA_PATH="${PROJECT_ROOT}/data/processed/generator/smiles.tsv"
if [[ ! -f "$DATA_PATH" ]] && [[ ! -d "${PROJECT_ROOT}/data/admet_group" ]]; then
  echo "Error: need data/processed/generator/smiles.tsv or data/admet_group."
  echo "Run: scripts/download_data.py and/or scripts/download_chembl_smiles.py"
  exit 1
fi
if [[ -f "$DATA_PATH" ]]; then
  echo "Using SMILES: $DATA_PATH"
else
  echo "Using ADMET fallback (aggregated train_val SMILES). For more data run: python scripts/download_chembl_smiles.py"
fi

# 2. Pretrain with production defaults + larger model (best-by-validity checkpointing)
echo ""
echo "--- Pretrain (30 epochs, 200k SMILES, batch 128, d_model=384, num_layers=8) ---"
python3 scripts/train_generator.py --stage pretrain --production --d-model 384 --num-layers 8
echo "Pretrain done. Best checkpoint (by validity) saved to checkpoints/generator."

# 3. RL with production defaults (best-by-reward checkpointing)
echo ""
echo "--- RL (15 epochs, batch 16) ---"
python3 scripts/train_generator.py --stage rl --production --resume checkpoints/generator
echo "RL done. Best checkpoint (by reward) saved to checkpoints/generator_rl."

echo ""
echo "=== Production training complete ==="
echo "Use checkpoints/generator_rl for the app and pipeline (best model)."
