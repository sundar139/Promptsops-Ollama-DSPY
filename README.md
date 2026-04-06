# Promptsops-Ollama-DSPY

Local-first PromptsOps template for building, evaluating, and regression-testing DSPy programs with Ollama.

[![CI](https://github.com/sundar139/Promptsops-Ollama-DSPY/actions/workflows/ci.yml/badge.svg)](https://github.com/sundar139/Promptsops-Ollama-DSPY/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

## Overview

Prompt changes often ship without the same quality controls used for application code. This repository provides a practical local workflow to compile a DSPy program, evaluate it on a benchmark, track score drift over time, and gate quality in CI.

It is designed for teams and individuals who want reproducible prompt iteration with minimal infrastructure: local Ollama models, deterministic scoring, optional LLM-as-judge checks, and optional OpenTelemetry tracing.

### Who This Is For

- ML engineers who want repeatable prompt optimization loops.
- MLOps engineers who want lightweight CI quality gates for LLM behavior.
- Recruiters and hiring managers evaluating practical ML infrastructure craftsmanship.

### Why Local DSPy + Ollama

- Run and iterate offline without cloud inference dependencies.
- Keep costs predictable while building evaluation discipline.
- Preserve a cloud-ready engineering pattern (artifacting, tests, CI, regression checks) in a local setup.

## Key Features

- DSPy QA program with local Ollama runtime.
- Optimization pipeline that compiles and saves reusable artifacts.
- Deterministic evaluation and optional LLM-as-judge scoring.
- Benchmark result persistence (`latest.json` + timestamped runs).
- Benchmark regression checker with optional fail threshold.
- Fast CI gate plus manual extended runtime checks.
- Optional OpenTelemetry tracing hooks.

## Architecture Overview

- High-level diagram and component map: [docs/architecture.md](docs/architecture.md)
- Benchmarking details and regression utility: [docs/benchmarking.md](docs/benchmarking.md)

## Quickstart

Prerequisites:
- Python 3.11+
- uv
- Ollama running locally

```sh
ollama pull llama3.2:3b
ollama pull llama3.2:1b

uv sync --group dev

uv run python scripts/healthcheck.py
uv run python scripts/optimize.py
uv run python scripts/run_eval.py
uv run python scripts/check_benchmark_regression.py
```

## Example Workflow

```sh
# 1) Confirm runtime prerequisites
uv run python scripts/healthcheck.py

# 2) Compile an artifact (optionally bounded)
uv run python scripts/optimize.py --max-train-examples 8 --max-bootstrapped-demos 4

# 3) Evaluate compiled artifact and write benchmark outputs
uv run python scripts/run_eval.py

# 4) Compare latest benchmark to prior run
uv run python scripts/check_benchmark_regression.py --max-regression 0.03

# 5) Run local quality gate
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts tests
uv run pytest -q -m "not slow" --cov=promptsops --cov-report=term-missing --cov-report=xml
```

## Repository Structure

```text
src/promptsops/           Core library (config, dataset, program, metrics, optimizer, tracing)
scripts/                  Executable entrypoints (healthcheck, optimize, eval, regression check)
tests/                    Unit, integration, and slow test suites
artifacts/                Compiled artifacts and benchmark result snapshots
docs/                     Focused architecture and benchmarking documentation
.github/workflows/ci.yml Fast CI gate + manual extended checks
```

## Evaluation and Benchmarking

- `scripts/run_eval.py` evaluates the compiled artifact on the dev split.
- Results are written to:
  - `artifacts/benchmark_results/latest.json`
  - `artifacts/benchmark_results/run_YYYYMMDDTHHMMSSZ.json`
- Regression comparison utility:

```sh
uv run python scripts/check_benchmark_regression.py
uv run python scripts/check_benchmark_regression.py --max-regression 0.03
```

For full behavior and options, see [docs/benchmarking.md](docs/benchmarking.md).

## CI and Test Strategy

Default CI (`.github/workflows/ci.yml`) is optimized for reliability and speed:
- Runs on push/PR for `main` and `master`.
- Executes lint, formatting check, type checking, and fast tests with coverage (`-m "not integration and not slow"`).
- Uploads `coverage.xml` as an artifact.

Manual extended checks are available via `workflow_dispatch`:
- Runtime healthcheck and Ollama model pull
- Bounded optimization
- Integration tests (`integration and not slow`)
- Benchmark regression check (optional threshold)
- Optional slow judge tests

Test marker conventions:
- `integration`: requires local Ollama runtime/models.
- `slow`: judge-heavy tests, excluded from default CI.

## Configuration / Environment Variables

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `GENERATOR_MODEL` (default: `llama3.2:3b`)
- `JUDGE_MODEL` (default: `llama3.2:1b`)
- `DSPY_TEMPERATURE` (default: `0.2`)
- `ENABLE_TRACING` (`1`/`true`/`yes` to enable tracing)
- `PHOENIX_ENDPOINT` (default: `http://localhost:6006/v1/traces`)

CI note:
- Set `HF_TOKEN` in GitHub Actions secrets for authenticated Hugging Face access and improved rate limits.

## Troubleshooting

Ollama unavailable:
- Verify server is running: `ollama list`
- Verify base URL: `OLLAMA_BASE_URL`
- Re-run healthcheck: `uv run python scripts/healthcheck.py`

Missing compiled artifact:
- Generate it with `uv run python scripts/optimize.py`

Slow or flaky benchmark pulls in CI:
- Configure `HF_TOKEN` secret in GitHub Actions.

Tracing is not visible:
- Set `ENABLE_TRACING=1`
- Ensure your OTLP endpoint is reachable (for example, Phoenix)

## Roadmap / Future Improvements

- Add richer benchmark suites beyond Tiny QA.
- Expand regression policies (for example, branch-aware thresholds).
- Add optional artifact versioning policy and release tagging discipline.
- Improve judge robustness with calibrated model choices and thresholds.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, validation commands, branch/PR expectations, and style guidance.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
