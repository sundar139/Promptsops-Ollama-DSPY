from __future__ import annotations

import argparse

from promptsops.optimizer import optimize_tinyqa_program


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and save an optimized DSPy artifact.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output path for compiled artifact JSON.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional cap on training examples used for optimization.",
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=8,
        help="Max demos BootstrapFewShot can compile.",
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=32,
        help="Max labeled demos for BootstrapFewShot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    optimize_tinyqa_program(
        output_path=args.output_path,
        max_train_examples=args.max_train_examples,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )
