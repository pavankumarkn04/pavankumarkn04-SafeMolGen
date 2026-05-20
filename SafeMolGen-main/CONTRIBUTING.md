# Contributing to SafeMolGen

Thanks for your interest. This project welcomes focused, well-tested contributions.

## Ground rules

- Discuss non-trivial changes in an issue before opening a PR.
- One logical change per PR. Keep diffs reviewable.
- No emojis in commits or code. Use imperative commit subjects (`Fix generator OOV handling`, not `Fixed ...`).
- Every change must keep `pytest` and the frontend `tsc -b && vite build` green.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Optional — for training and heavy evaluation:

```bash
pip install -r requirements-e2e.txt
```

## Running tests

```bash
PYTHONPATH=. pytest tests/ -v
cd frontend && npm run build
```

## Code style

- Python: PEP8, type hints on public APIs, `ruff` clean.
- TypeScript: strict mode, no implicit `any`, prefer composition over inheritance.
- Docstrings explain *why*, not *what*.

## Commit convention

```
<area>: <concise imperative subject>

Optional body with context, trade-offs, and links to issues.
```

Areas: `generator`, `oracle`, `admet`, `reranker`, `backend`, `frontend`, `docs`, `ci`, `build`.

## Pull requests

1. Branch from `main` with a descriptive name (`fix/oracle-phase-calibration`).
2. Rebase (not merge) to keep history linear.
3. Fill out the PR template. Include before/after behavior, test evidence, and any schema changes.

## Reporting bugs

Open an issue with:

- Reproduction steps (minimal example preferred)
- Expected vs actual behavior
- Environment (OS, Python, PyTorch, RDKit versions)
- Relevant logs or tracebacks

## Security

Do not file public issues for vulnerabilities. Contact the maintainer directly.
