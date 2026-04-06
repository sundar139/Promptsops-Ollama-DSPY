# Contributing

Thanks for contributing. Keep changes focused, testable, and easy to review.

## Local Setup

1. Install prerequisites:
- Python 3.11+
- uv
- Ollama

2. Pull required models:

```sh
ollama pull llama3.2:3b
ollama pull llama3.2:1b
```

3. Install dependencies:

```sh
uv sync --group dev
```

4. Verify local runtime:

```sh
uv run python scripts/healthcheck.py
```

## Development Workflow

1. Create a branch from `main`.
2. Keep commits small and logically grouped.
3. Run checks before opening a PR.
4. Open a PR with clear scope and validation evidence.

## Required Checks Before PR

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts tests
uv run pytest -q -m "not integration and not slow" --cov=promptsops --cov-report=term-missing --cov-report=xml
```

Optional extended checks:

```sh
uv run python scripts/optimize.py --max-train-examples 4 --max-bootstrapped-demos 2
uv run pytest -q -m "integration and not slow"
uv run pytest -q -m "slow"
uv run python scripts/run_eval.py
uv run python scripts/check_benchmark_regression.py --max-regression 0.03
```

## Branch and PR Expectations

- Preferred target branch: `main`.
- CI still validates both `main` and `master` for compatibility.
- PR descriptions should include:
  - what changed
  - why it changed
  - exact commands run locally

## Style Expectations

- Follow existing naming and structure.
- Keep implementation practical; avoid speculative abstractions.
- Add comments only when behavior is non-obvious.
- Treat prompts and benchmark behavior as production code: changes need tests or a clear rationale.
