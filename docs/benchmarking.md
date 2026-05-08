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

## Generate Baseline vs Compiled Lift Table

To compute baseline (uncompiled) and compiled QA accuracy and update the README comparison table:

```sh
uv run python scripts/update_lift_table.py --sample-size 20 --write-readme
```

This command evaluates both variants on the same dev slice and rewrites the section between
`LIFT_TABLE_START` and `LIFT_TABLE_END` markers in `README.md`.
