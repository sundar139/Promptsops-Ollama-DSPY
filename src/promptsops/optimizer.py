from __future__ import annotations

from pathlib import Path

import dspy

from .config import configure_lm, load_runtime_config
from .dataset import load_tinyqa_examples
from .healthcheck import assert_ollama_ready
from .metrics import deterministic_metric
from .program import TinyQAProgram

ARTIFACT_PATH = Path("artifacts/compiled_program.json")


def optimize_tinyqa_program(
    output_path: str | Path | None = None,
    max_train_examples: int | None = None,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 32,
) -> str:
    if max_train_examples is not None and max_train_examples <= 0:
        raise ValueError(f"max_train_examples must be > 0. Got: {max_train_examples}")
    if max_bootstrapped_demos <= 0:
        raise ValueError(f"max_bootstrapped_demos must be > 0. Got: {max_bootstrapped_demos}")
    if max_labeled_demos <= 0:
        raise ValueError(f"max_labeled_demos must be > 0. Got: {max_labeled_demos}")

    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    train, _ = load_tinyqa_examples()

    if max_train_examples is not None:
        train = train[:max_train_examples]
    if not train:
        raise ValueError("No training examples available for optimization.")

    def metric_fn(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: dspy.Trace | None = None,
    ) -> float:
        del trace
        return deterministic_metric(example, prediction).score

    optimizer = dspy.BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=min(max_bootstrapped_demos, len(train)),
        max_labeled_demos=min(max_labeled_demos, len(train)),
    )

    program = TinyQAProgram()
    compiled = optimizer.compile(program, trainset=train)

    save_path = Path(output_path) if output_path is not None else ARTIFACT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(save_path), save_program=False)
    print(f"Saved compiled artifact to: {save_path}")
    return str(save_path)
