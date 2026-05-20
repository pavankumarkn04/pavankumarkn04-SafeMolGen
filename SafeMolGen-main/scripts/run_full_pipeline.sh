#!/usr/bin/env bash
# Run full pipeline in order: ADMET data -> ADMET train -> Clinical CSV -> Oracle train -> Generator pretrain -> (optional) Generator RL -> run_pipeline.
# Usage: from project root, with venv activated and PYTHONPATH=.
#   PYTHONPATH=. bash scripts/run_full_pipeline.sh

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

echo "=== 1. Prepare ADMET data from admet_group ==="
python3 scripts/prepare_admet_from_admet_group.py

echo "=== 2. Preprocess ADMET to .pt ==="
python3 scripts/preprocess_data.py

echo "=== 3. Train ADMET ==="
python3 scripts/train_admet.py

echo "=== 4. Prepare clinical_trials.csv ==="
python3 scripts/prepare_clinical_data.py

echo "=== 5. Train Oracle ==="
python3 scripts/train_oracle.py

echo "=== 6. Generator pretrain (30 epochs) ==="
python3 scripts/train_generator.py --stage pretrain --epochs 30 --limit 50000 --batch-size 64

echo "=== 7. Post-pretrain eval ==="
python3 scripts/generate_samples.py --model checkpoints/generator --n 500 --temperature 0.8 --top-k 40

echo "=== 8. Generator RL (optional) ==="
python3 scripts/train_generator.py --stage rl --resume checkpoints/generator --epochs 10 --w-validity 0.75 --w-diversity 0.1

echo "=== 9. Run pipeline (design_molecule) ==="
python3 scripts/run_pipeline.py --out outputs/design_result.json

echo "Done. Run the app: python scripts/run_app.py (backend) and cd frontend && npm run dev (frontend)"
