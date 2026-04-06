# promptsops-ollama-dspy

Local PromptsOps template for DSPy + Ollama.
The repo evaluates and optimizes a tiny QA program, saves a compiled artifact,
enforces quality gates, and can optionally emit OpenTelemetry traces.

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

Default CI excludes slow judge tests with -m "not slow".

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

## CI Behavior and Supported Branches

Workflow file: .github/workflows/ci.yml

CI runs on:
- push to main and master
- pull requests targeting main and master
- manual trigger via workflow_dispatch

CI stages:
1. Checkout
2. Setup uv
3. Sync dependencies
4. Install/start Ollama
5. Pull required models
6. Run healthcheck
7. Optimize program
8. Run lint
9. Run type checks
10. Run tests with coverage

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
