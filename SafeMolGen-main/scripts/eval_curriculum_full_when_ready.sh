#!/usr/bin/env bash
# Run after full curriculum finishes. Eval base vs generator_curriculum_full.
#   PYTHONPATH=. bash scripts/eval_curriculum_full_when_ready.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

FULL="checkpoints/generator_curriculum_full/best"
if [ ! -f "$FULL/model.pt" ]; then
  echo "Full curriculum not ready: $FULL/model.pt missing. Wait for run_option_b_curriculum.py to finish."
  exit 1
fi
echo "Eval: Base vs curriculum_full (n=150)"
PYTHONPATH=. python3 scripts/eval_generator_oracle.py --base checkpoints/generator/best --rl "$FULL" --target-phase 0.6 --n 150
