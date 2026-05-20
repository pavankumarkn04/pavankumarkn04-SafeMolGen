#!/usr/bin/env bash
# Option B full pipeline: curate high-oracle SMILES -> target-condition pretrain (0.6) -> evaluate.
# Run from project root with PYTHONPATH set.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

CURATED_TSV="${CURATED_TSV:-data/processed/generator/smiles_oracle_curated.tsv}"
TARGET_PHASE="${TARGET_PHASE:-0.6}"
TOP_PCT="${TOP_PCT:-20}"
LIMIT="${LIMIT:-200000}"
EPOCHS="${EPOCHS:-30}"
EVAL_N="${EVAL_N:-150}"

echo "=== Step 1: Build Oracle-curated SMILES (top ${TOP_PCT}%, limit ${LIMIT}) ==="
python scripts/build_oracle_curated_smiles.py \
  --data data/processed/generator/smiles.tsv \
  --out "$CURATED_TSV" \
  --limit "$LIMIT" \
  --top-pct "$TOP_PCT"

echo ""
echo "=== Step 2: Target-condition pretrain (target_phase=${TARGET_PHASE}, epochs=${EPOCHS}) ==="
python scripts/train_generator.py --stage pretrain \
  --data "$CURATED_TSV" \
  --use-target-condition \
  --target-phase "$TARGET_PHASE" \
  --limit "$LIMIT" \
  --epochs "$EPOCHS" \
  --out checkpoints/generator

echo ""
echo "=== Step 3: Evaluate generator with Oracle (n=${EVAL_N}) ==="
python scripts/eval_generator_oracle.py --n "$EVAL_N" --base checkpoints/generator

echo ""
echo "Done. Generator checkpoint: checkpoints/generator (or checkpoints/generator/best)."
