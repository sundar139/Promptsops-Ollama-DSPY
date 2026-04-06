from __future__ import annotations

from promptsops.config import configure_lm
from promptsops.dataset import load_tinyqa_examples
from promptsops.metrics import deterministic_metric, EvalResult
from promptsops.program import TinyQAProgram


def evaluate_program() -> float:
    configure_lm()
    _, dev = load_tinyqa_examples()
    program = TinyQAProgram()

    scores: list[float] = []
    results: list[EvalResult] = []
    for ex in dev:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        scores.append(result.score)
        results.append(result)

    avg = sum(scores) / len(scores)
    return avg


if __name__ == "__main__":
    avg_score = evaluate_program()
    print(f"Average deterministic score: {avg_score:.3f}")
