#!/usr/bin/env bash
# Run the design pipeline from CLI with live monitoring.
# Usage:
#   ./scripts/run_monitor.sh
#   ./scripts/run_monitor.sh --max-iterations 5
#   ./scripts/run_monitor.sh --out outputs/result.json
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
exec python3 scripts/run_pipeline.py "$@"
