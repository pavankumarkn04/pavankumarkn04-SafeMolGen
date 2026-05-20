#!/usr/bin/env bash
# Run the design pipeline with all possible solutions enabled (plan 5.1, 5.2, 4.3):
#   - Multiple restarts (best of N runs)
#   - Diversity selection (avoid similar molecules)
#   - Two-stage reranker when checkpoint exists (pre-filter candidates)
# Usage:
#   PYTHONPATH=. bash scripts/run_pipeline_all_solutions.sh
#   PYTHONPATH=. bash scripts/run_pipeline_all_solutions.sh --out outputs/result.json --max-iterations 8
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

exec python3 scripts/run_pipeline.py --all-solutions "$@"
