#!/usr/bin/env bash
# Curriculum RL: start from Option B; phase 1 preserves validity, phase 2 pushes oracle.
# Next-best option when RL and base are similar (eval suggests this script).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

BASE="${1:-checkpoints/generator/best}"
OUT="${2:-checkpoints/generator_rl_curriculum}"
EPOCHS_P1="${EPOCHS_P1:-5}"
EPOCHS_P2="${EPOCHS_P2:-8}"
BATCH="${BATCH:-12}"

if [ ! -f "$BASE/model.pt" ]; then
  echo "Base checkpoint not found at $BASE. Run Option B first: python scripts/run_option_b_full.py"
  exit 1
fi

echo "=== Phase 1: RL from Option B, low oracle weight (preserve validity) ==="
python3 scripts/train_generator.py --stage rl --resume "$BASE" --out "${OUT}_phase1" \
  --w-oracle 0.15 --epochs "$EPOCHS_P1" --batch-size "$BATCH"

echo ""
echo "=== Phase 2: RL from Phase 1 best, moderate oracle weight (avoid validity collapse) ==="
python3 scripts/train_generator.py --stage rl --resume "${OUT}_phase1/best" --out "$OUT" \
  --w-oracle 0.25 --epochs "$EPOCHS_P2" --batch-size "$BATCH" --batch-normalize-oracle

echo ""
echo "Done. Evaluate: PYTHONPATH=. python3 scripts/eval_generator_oracle.py --base checkpoints/generator/best --rl $OUT/best --target-phase 0.6 --n 150"
