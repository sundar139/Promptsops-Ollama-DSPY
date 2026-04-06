import pytest

from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import deterministic_metric

THRESHOLD = 0.60  # Conservative; measured baseline ~0.73


@pytest.mark.integration
def test_tinyqa_compiled_program_quality():
    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    _, dev = load_tinyqa_examples()
    program = load_compiled_program()

    scores: list[float] = []
    for ex in dev:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        scores.append(result.score)

    avg_score = sum(scores) / len(scores)
    assert avg_score >= THRESHOLD, (
        f"Score {avg_score:.3f} is below threshold {THRESHOLD}. "
        "Run optimize.py to regenerate the artifact."
    )
