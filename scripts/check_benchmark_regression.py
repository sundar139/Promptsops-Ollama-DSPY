from __future__ import annotations

import argparse

from promptsops.benchmark_regression import compare_latest_to_previous, evaluate_regression_gate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latest benchmark score against the most recent prior benchmark run."
    )
    parser.add_argument(
        "--results-dir",
        default="artifacts/benchmark_results",
        help="Directory containing latest.json and run_*.json benchmark files.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=None,
        help="Optional max tolerated score drop before failing (e.g. 0.03).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    comparison = compare_latest_to_previous(results_dir=args.results_dir)

    print(f"Latest benchmark: {comparison.latest.path} (score={comparison.latest.score:.3f})")
    if comparison.baseline is not None:
        print(
            f"Baseline benchmark: {comparison.baseline.path} "
            f"(score={comparison.baseline.score:.3f})"
        )

    ok, message = evaluate_regression_gate(
        comparison=comparison,
        max_regression=args.max_regression,
    )
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
