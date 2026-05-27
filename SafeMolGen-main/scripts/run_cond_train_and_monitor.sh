#!/usr/bin/env bash
# Run conditioned pretrain then RL with live output. Kill with Ctrl+C or pkill -f run_cond_train_and_monitor
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p logs

echo "========== CONDITIONED PRETRAIN (10 ep, 50k) =========="
python3 -u scripts/train_generator.py --stage pretrain --epochs 10 --limit 50000 2>&1 | tee logs/cond_pretrain.log

echo ""
echo "========== PRETRAIN DONE. STARTING RL (5 ep) =========="
python3 -u scripts/train_generator.py --stage rl --resume checkpoints/generator --epochs 5 2>&1 | tee logs/cond_rl.log

echo ""
echo "========== ALL DONE. Checkpoints: generator (cond_dim=25), generator_rl =========="
