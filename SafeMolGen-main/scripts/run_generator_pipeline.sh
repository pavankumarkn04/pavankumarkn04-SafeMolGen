#!/usr/bin/env bash
# Run generator pipeline: pretrain -> check validity -> optional RL.
# Usage: from project root, with venv/conda activated and deps installed:
#   PYTHONPATH=. bash scripts/run_generator_pipeline.sh
# Or: PYTHONPATH=. python3 scripts/run_generator_pipeline.py

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "=== Step 1: Generator data ==="
if [ ! -f "data/processed/generator/smiles.tsv" ]; then
  echo "data/processed/generator/smiles.tsv not found. Run: python3 scripts/download_chembl_smiles.py"
  echo "Or ensure data/admet_group/*/train_val.csv exist for fallback."
  exit 1
fi
echo "Found generator SMILES. Line count: $(wc -l < data/processed/generator/smiles.tsv)"

echo ""
echo "=== Step 2: Pretrain (3 epochs, 20k SMILES) ==="
python3 scripts/train_generator.py --stage pretrain --epochs 3 --limit 20000 --batch-size 64

echo ""
echo "=== Step 3: Check validity (generate 100 samples) ==="
python3 scripts/generate_samples.py --model checkpoints/generator --n 100 --top-k 40

VALIDITY_LINE=$(python3 scripts/generate_samples.py --model checkpoints/generator --n 50 --top-k 40 2>/dev/null | grep "Valid:" || true)
echo "Validity check: $VALIDITY_LINE"

echo ""
echo "=== Optional: RL (run only if validity > 0%) ==="
echo "If validity above is > 0%, run:"
echo "  PYTHONPATH=. python3 scripts/train_generator.py --stage rl --resume checkpoints/generator"
echo "Done."
