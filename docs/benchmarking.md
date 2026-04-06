# Benchmarking

This project stores deterministic benchmark runs in:

- artifacts/benchmark_results/latest.json
- artifacts/benchmark_results/run_YYYYMMDDTHHMMSSZ.json

## Run a benchmark

```sh
uv run python scripts/run_eval.py
```

## Compare with previous run

```sh
uv run python scripts/check_benchmark_regression.py
```

Example with threshold enforcement:

```sh
uv run python scripts/check_benchmark_regression.py --max-regression 0.03
```

If your repo layout uses `benchmarks/latest.json` instead, point the utility to that directory:

```sh
uv run python scripts/check_benchmark_regression.py --results-dir benchmarks
```

Behavior:
- If no prior run exists, the script prints a skip message and exits successfully.
- If a prior run exists, it reports baseline score, latest score, and delta.
- With --max-regression, the command fails only when the score drop exceeds the threshold.

This keeps regression tracking deterministic and lightweight for local development and optional CI usage.
