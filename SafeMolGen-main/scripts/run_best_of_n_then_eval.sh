#!/usr/bin/env bash
# Best-of-N fine-tune from Option B, then eval vs base. Stable supervised alternative when RL matches but doesn't beat base.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

BASE="${1:-checkpoints/generator/best}"
OUT="${2:-checkpoints/generator_best_of_n}"
EPOCHS="${EPOCHS:-15}"
N="${N:-150}"

if [ ! -f "$BASE/model.pt" ]; then
  echo "Base checkpoint not found at $BASE. Run Option B first: python scripts/run_option_b_full.py"
  exit 1
fi

echo "=== Best-of-N fine-tune (resume=$BASE, epochs=$EPOCHS) ==="
python3 scripts/train_best_of_n.py --resume "$BASE" --out "$OUT" --epochs "$EPOCHS" --batch-size 64

echo ""
echo "=== Eval: Base vs Best-of-N (target-phase 0.6, n=$N) ==="
PYTHONPATH=. python3 scripts/eval_generator_oracle.py --base "$BASE" --rl "$OUT" --target-phase 0.6 --n "$N"
