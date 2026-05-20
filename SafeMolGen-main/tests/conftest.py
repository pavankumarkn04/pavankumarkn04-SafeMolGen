"""Pytest configuration and shared fixtures."""

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PORT = 5173
E2E_BASE_URL = os.environ.get("E2E_APP_URL", f"http://localhost:{FRONTEND_PORT}")


def _wait_for_url(url: str, timeout: int = 60, interval: float = 1.0) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(interval)
    return False


@pytest.fixture(scope="session")
def _frontend_process():
    """Start frontend dev server for e2e tests when E2E_APP_URL is not set."""
    if os.environ.get("E2E_APP_URL"):
        yield None
        return
    frontend_dir = PROJECT_ROOT / "frontend"
    if not (frontend_dir / "package.json").exists():
        yield None
        return
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ},
    )
    try:
        if not _wait_for_url(f"http://127.0.0.1:{FRONTEND_PORT}", timeout=60):
            proc.terminate()
            proc.wait(timeout=5)
            pytest.skip(
                f"Frontend did not become ready at http://localhost:{FRONTEND_PORT}."
            )
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
