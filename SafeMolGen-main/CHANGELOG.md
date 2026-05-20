# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete training and evaluation stack for ADMET, Oracle, Generator (pretrain + RL), and Reranker.
- `ensure_target` multi-strategy design mode (single → restarts → diversity-restarts cascade).
- Structural alerts subsystem with Hamburg PAINS/SMARTS ingestion.
- Server-Sent Events streaming endpoint for iterative design (`POST /api/v1/design/stream`).
- Centralized logging via `utils.logging_config` and runtime metrics in design traces.
- GitHub Actions CI: Ruff, pytest, frontend typecheck/build.
- MIT License, Contribution guide, architecture diagrams, API reference.

### Changed
- Repository restructured: guides, reports, and architecture docs moved under `docs/`.
- Large binary artifacts tracked with Git LFS (`*.pt`, `*.pth`, `*.docx`).
- `.gitignore` hardened against local `.venv`, `node_modules`, data dumps, and editor artifacts.

### Removed
- Unused Analyze/Compare UI pages (superseded by integrated Generate workflow).

## [0.1.0] - 2026-02-01

### Added
- Initial public release with ADMET predictor, DrugOracle, SafeMolGen generator,
  FastAPI backend, and React + Chakra UI frontend.
