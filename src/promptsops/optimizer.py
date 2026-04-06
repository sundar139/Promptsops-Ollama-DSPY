from __future__ import annotations

from pathlib import Path

import dspy

from .config import configure_lm, load_runtime_config
from .dataset import load_tinyqa_examples
from .healthcheck import assert_ollama_ready
from .metrics import deterministic_metric
from .program import TinyQAProgram

ARTIFACT_PATH = Path("artifacts/compiled_program.json")


def optimize_tinyqa_program(output_path: str | None = None) -> str:
    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    train, _ = load_tinyqa_examples()

    def metric_fn(
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: dspy.Trace | None = None,
    ) -> float:
        del trace
        return deterministic_metric(example, prediction).score

    optimizer = dspy.BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=8,
        max_labeled_demos=32,
    )

    program = TinyQAProgram()
    compiled = optimizer.compile(program, trainset=train)

    save_path = Path(output_path) if output_path else ARTIFACT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(save_path), save_program=False)
    print(f"Saved compiled artifact to: {save_path}")
    return str(save_path)
