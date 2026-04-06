# promptsops-ollama-dspy

[![CI](https://github.com/sundar139/Promptsops-Ollama-DSPY/actions/workflows/ci.yml/badge.svg)](https://github.com/sundar139/Promptsops-Ollama-DSPY/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

Local PromptsOps template for DSPy + Ollama.
The repo evaluates and optimizes a tiny QA program, saves a compiled artifact,
enforces quality gates, and can optionally emit OpenTelemetry traces.

Start here:
- [Architecture](docs/architecture.md)
- [Benchmarking](docs/benchmarking.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## What Was Added In This Session

- Fast CI gate now runs lint, type checks, and fast tests independent of heavy optimization runtime.
- Manual extended checks run healthcheck, bounded optimization, integration tests, benchmark regression checks, and optional slow judge tests.
- Benchmark regression utility is now part of standard workflow:
  - `uv run python scripts/check_benchmark_regression.py`
  - optional fail gate: `--max-regression 0.03`
- Optimization is now bounded/configurable via `scripts/optimize.py` flags:
  - `--max-train-examples`
  - `--max-bootstrapped-demos`
  - `--max-labeled-demos`
- CI now reads `HF_TOKEN` from GitHub Actions secrets when configured.
- Collaboration hygiene added: issue templates, PR template, CONTRIBUTING, CHANGELOG, docs/architecture, docs/benchmarking.

## Quickstart

Prerequisites:
- Ollama installed and running
- uv installed
- Python 3.11+

Commands:

```sh
ollama pull llama3.2:3b
ollama pull llama3.2:1b

uv sync --group dev

uv run python scripts/healthcheck.py
uv run python scripts/optimize.py
uv run python scripts/run_eval.py
uv run python scripts/check_benchmark_regression.py
```

The evaluation command writes benchmark artifacts to artifacts/benchmark_results/.

## Validation

Validation covers:
- Ollama connectivity and required model availability
- Program optimization and compiled artifact generation
- Linting and formatting checks
- Static typing checks
- Deterministic and integration pytest checks
- Coverage reporting for the promptsops package

Default CI runs fast unit-focused checks with `-m "not integration and not slow"`.
Integration and optimization checks are available via manual extended workflow dispatch.

## Development Checks

Run these locally before pushing:

```sh
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts tests
uv run pytest -q -m "not slow" --cov=promptsops --cov-report=term-missing --cov-report=xml
```

Run the slow judge test explicitly:

```sh
uv run pytest -q -m "slow"
```

## Test Scope

| Marker / command | Scope | Default CI |
| --- | --- | --- |
| `-m "not integration and not slow"` | Fast unit-focused checks | Yes |
| `-m "not slow"` | Unit + integration tests excluding judge-heavy checks | No (recommended local gate) |
| `-m "slow"` | LLM-as-judge checks | No (manual only) |
| `-m "not integration"` | Pure local unit-only checks | No |

Coverage target is intentionally pragmatic (`fail_under = 60`) and currently sits above that floor. The focus is confidence and regression protection, not artificial 100% targets.

## Benchmark Regression Tracking

`run_eval.py` writes benchmark JSON snapshots to `artifacts/benchmark_results/`.
Use the regression utility to compare `latest.json` to the most recent prior run.

```sh
uv run python scripts/check_benchmark_regression.py
```

Optional threshold gate (fails only on significant regressions):

```sh
uv run python scripts/check_benchmark_regression.py --max-regression 0.03
```

## CI Behavior and Supported Branches

Workflow file: .github/workflows/ci.yml

CI runs on:
- push to main and master
- pull requests targeting main and master
- manual trigger via workflow_dispatch

Manual workflow inputs (for bounded extended checks):
- `run_extended_checks`
- `run_slow_tests`
- `benchmark_max_regression`
- `optimize_max_train_examples`
- `optimize_max_bootstrapped_demos`

CI stages:
1. Checkout
2. Setup uv
3. Sync dependencies
4. Run lint
5. Run type checks
6. Run fast tests with coverage
7. Upload coverage artifact

Manually triggered extended checks run runtime healthcheck, bounded optimization, integration tests, benchmark regression checks, and optional slow judge tests.

## Branch Policy

- `main` is the intended long-term default branch.
- `master` remains fully supported in CI for compatibility.
- Pull requests are expected to target `main` unless a migration/backport scenario requires otherwise.

## Test Markers

- integration: tests that require a running local Ollama with models
- slow: judge-style tests that are intentionally slower and excluded in default CI

Examples:

```sh
uv run pytest -q -m "not integration"
uv run pytest -q -m "integration and not slow"
uv run pytest -q -m "slow"
```

## Benchmark Result Format

Each run_eval invocation writes:
- Timestamp (UTC)
- Generator model name
- Train split size
- Dev split size
- Deterministic average score

Files:
- artifacts/benchmark_results/latest.json
- artifacts/benchmark_results/run_YYYYMMDDTHHMMSSZ.json

This enables simple local run-to-run comparison without adding extra services.

## Troubleshooting

Ollama unavailable:
- Confirm server is running: ollama list
- Check OLLAMA_BASE_URL (default http://localhost:11434)
- Run healthcheck: uv run python scripts/healthcheck.py

Missing model error:
- Pull required models:
  - ollama pull llama3.2:3b
  - ollama pull llama3.2:1b
- Or set GENERATOR_MODEL and JUDGE_MODEL to models you have locally.

HF dataset rate limits in CI:
- Add `HF_TOKEN` as a GitHub Actions secret and it will be picked up automatically by the workflow.

Artifact missing:
- Generate compiled artifact: uv run python scripts/optimize.py

Tracing optional setup:
- Set ENABLE_TRACING=1 and optionally PHOENIX_ENDPOINT
- Start Phoenix separately (for example via Docker)

## Environment Variables

- OLLAMA_BASE_URL: Ollama API base URL
- GENERATOR_MODEL: model used for program generation
- JUDGE_MODEL: model used for optional LLM-as-judge
- DSPY_TEMPERATURE: default DSPy LM temperature
- ENABLE_TRACING: set to 1/true/yes to enable tracing
- PHOENIX_ENDPOINT: OTLP traces endpoint

## Repository Layout

- src/promptsops/: core library modules
- scripts/: executable entry points
- tests/: unit/integration/slow tests
- artifacts/: generated compiled program and benchmark outputs
- docs/: focused architecture and benchmarking references
