#!/usr/bin/env bash
# RL starting from Option B (plan 1.4): use generator/best as resume so RL nudges from high-oracle prior.
# Usage: from project root, PYTHONPATH=. bash scripts/run_rl_from_option_b.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

BASE="${1:-checkpoints/generator/best}"
OUT="${2:-checkpoints/generator_rl}"
W_ORACLE="${W_ORACLE:-0.25}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-12}"

if [ ! -f "$BASE/model.pt" ]; then
  echo "Option B checkpoint not found at $BASE. Run Option B first: python scripts/run_option_b_full.py"
  exit 1
fi

echo "=== RL from Option B (resume=$BASE, w_oracle=$W_ORACLE, epochs=$EPOCHS) ==="
python3 scripts/train_generator.py --stage rl \
  --resume "$BASE" \
  --out "$OUT" \
  --w-oracle "$W_ORACLE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH"

echo ""
echo "Evaluate: PYTHONPATH=. python3 scripts/eval_generator_oracle.py --base $OUT/best --rl $OUT --target-phase 0.6 --n 150"
