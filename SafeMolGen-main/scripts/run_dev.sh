#!/usr/bin/env bash
# Run both FastAPI backend and React frontend with one command.
# Stop with Ctrl+C; the backend will be killed automatically.
# Open http://localhost:5173 in your browser (not 8000 — that's the API only).

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

if [ ! -f frontend/node_modules/.bin/vite ] && [ ! -f frontend/node_modules/vite/bin/vite.js ]; then
  echo "Installing frontend dependencies (first run may take a minute)..."
  (cd frontend && npm install)
fi

echo "Starting backend on http://127.0.0.1:8000 ..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!
trap "kill $UVICORN_PID 2>/dev/null; exit 0" EXIT INT TERM

sleep 2
echo "Starting frontend — open http://localhost:5173 in your browser"
cd frontend && npm run dev
