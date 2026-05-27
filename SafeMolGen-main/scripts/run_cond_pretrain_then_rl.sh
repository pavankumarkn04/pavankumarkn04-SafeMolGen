#!/usr/bin/env bash
# Run conditioned pretrain, then RL. Usage: ./scripts/run_cond_pretrain_then_rl.sh
# Or: bash scripts/run_cond_pretrain_then_rl.sh
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PRETRAIN_PID=""
run_pretrain() {
    python3 -u scripts/train_generator.py --stage pretrain --epochs 10 --limit 50000 2>&1 | tee logs/cond_pretrain.log
}
run_rl() {
    echo "Pretrain finished. Starting RL..."
    python3 -u scripts/train_generator.py --stage rl --resume checkpoints/generator --epochs 5 2>&1 | tee logs/cond_rl.log
    echo "RL done. Check checkpoints/generator_rl (cond_dim=25)."
}
# If checkpoints/generator has cond_dim in config and is recent, skip pretrain and run RL only
if [[ -f checkpoints/generator/model.pt ]]; then
    if python3 -c "
import torch
d = torch.load('checkpoints/generator/model.pt', map_location='cpu', weights_only=False)
c = d.get('config', {})
exit(0 if c.get('cond_dim', 0) == 25 else 1)
" 2>/dev/null; then
        echo "Found conditioned checkpoint at checkpoints/generator (cond_dim=25). Running RL only."
        run_rl
        exit 0
    fi
fi
echo "Starting conditioned pretrain (10 ep, 50k) -> checkpoints/generator"
run_pretrain
run_rl
