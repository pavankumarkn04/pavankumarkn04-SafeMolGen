#!/usr/bin/env bash
# Option B: Continuous monitor with progress bar. Run from project root:
#   ./scripts/watch_cond_training.sh
# Or: python3 scripts/watch_cond_training.py
# Status is written to logs/cond_monitor_status.txt for external monitoring.
cd "$(dirname "$0")/.."
exec python3 scripts/watch_cond_training.py
