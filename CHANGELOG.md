# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added
- Placeholder section for upcoming incremental improvements.

### Changed
- CI default quality job now runs lint, type checks, and fast non-integration tests before any heavy runtime tasks.
- Optimization in manual extended checks now supports bounded execution via train/demo caps.
- Added `HF_TOKEN` workflow env wiring to support authenticated Hugging Face dataset access in Actions.
- Marked artifact-existence test as integration so fast CI is no longer blocked on generated artifacts.

## [Stabilized Baseline - 2026-04-06]

### Added
- Runtime preflight checks for Ollama availability and required models.
- Deterministic benchmark result persistence (`latest.json` plus timestamped snapshots).
- Quality tooling and visible local/CI checks (Ruff, mypy, pytest-cov).
- Test marker strategy for `integration` and `slow` checks.

### Changed
- CI now supports both `main` and `master`, plus manual trigger support.
- README reorganized for practical runbooks (quickstart, validation, troubleshooting).
- Repository hygiene improved with editor settings, ignore rules, and benchmark artifact handling.

### Notes
- This baseline is the first repo state intended as a reusable engineering template.
