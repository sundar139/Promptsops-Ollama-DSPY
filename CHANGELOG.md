# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added
- Placeholder section for upcoming incremental improvements.

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
