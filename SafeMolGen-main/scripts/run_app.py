#!/usr/bin/env python3
"""Run the SafeMolGen-DrugOracle FastAPI backend."""

import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Run FastAPI with uvicorn
    os.system("uvicorn backend.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
