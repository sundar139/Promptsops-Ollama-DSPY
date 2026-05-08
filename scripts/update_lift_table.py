from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, Sequence

import dspy

from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import deterministic_metric
from promptsops.program import TinyQAProgram

START_MARKER = "<!-- LIFT_TABLE_START -->"
END_MARKER = "<!-- LIFT_TABLE_END -->"


class _QAProgram(Protocol):
    def __call__(self, *, context: str, question: str) -> dspy.Prediction: ...


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute baseline vs compiled QA accuracy and optionally update README table."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of dev examples to score (default: 20; uses full dev split if smaller).",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to README file containing lift table markers.",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Update the README section between LIFT_TABLE markers.",
    )
    return parser.parse_args()


def _score_program(program: _QAProgram, dev_examples: Sequence[dspy.Example]) -> float:
    if not dev_examples:
        raise ValueError("No dev examples available for evaluation.")

    scores: list[float] = []
    for ex in dev_examples:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        scores.append(result.score)

    return sum(scores) / len(scores)


def _build_table_block(baseline_score: float, compiled_score: float) -> str:
    delta = compiled_score - baseline_score
    return "\n".join(
        [
            "| Variant | QA Accuracy | Delta vs Baseline |",
            "| --- | ---: | ---: |",
            f"| Baseline QA program (before DSPy compile) | {baseline_score:.3f} | +0.000 |",
            f"| Compiled DSPy program (after optimization) | {compiled_score:.3f} | {delta:+.3f} |",
            "",
            f"Delta calculation: ${compiled_score:.3f} - {baseline_score:.3f} = {delta:+.3f}$.",
        ]
    )


def _replace_lift_section(readme_text: str, block: str) -> str:
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("README is missing LIFT_TABLE_START/LIFT_TABLE_END markers.")

    content_start = start + len(START_MARKER)
    replacement = f"\n{block}\n"
    return readme_text[:content_start] + replacement + readme_text[end:]


def main() -> None:
    args = _parse_args()
    if args.sample_size <= 0:
        raise ValueError(f"sample-size must be > 0. Got: {args.sample_size}")

    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)

    _, dev = load_tinyqa_examples()
    eval_examples = dev[: args.sample_size]

    baseline_program = TinyQAProgram()
    compiled_program = load_compiled_program()

    baseline_score = _score_program(baseline_program, eval_examples)
    compiled_score = _score_program(compiled_program, eval_examples)
    block = _build_table_block(baseline_score=baseline_score, compiled_score=compiled_score)

    print("Computed lift table:\n")
    print(block)

    if args.write_readme:
        readme_path = Path(args.readme_path)
        existing = readme_path.read_text(encoding="utf-8")
        updated = _replace_lift_section(existing, block)
        readme_path.write_text(updated, encoding="utf-8")
        print(f"\nUpdated README lift section: {readme_path}")


if __name__ == "__main__":
    main()
