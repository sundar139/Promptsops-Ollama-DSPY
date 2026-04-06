from __future__ import annotations

from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import deterministic_metric
from promptsops.program import TinyQAProgram


def evaluate_program() -> float:
    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    _, dev = load_tinyqa_examples()
    program = TinyQAProgram()

    scores: list[float] = []
    for ex in dev:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        scores.append(result.score)

    avg = sum(scores) / len(scores)
    return avg


if __name__ == "__main__":
    avg_score = evaluate_program()
    print(f"Average deterministic score: {avg_score:.3f}")
