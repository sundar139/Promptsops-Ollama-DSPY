from promptsops.config import configure_lm
from promptsops.dataset import load_tinyqa_examples
from promptsops.metrics import deterministic_metric
from promptsops.artifacts import load_compiled_program
from promptsops.tracing import is_tracing_enabled, start_tracing


def main() -> None:
    if is_tracing_enabled():
        start_tracing()
        print("Tracing enabled — exporting spans to Phoenix")

    configure_lm()
    _, dev = load_tinyqa_examples()
    program = load_compiled_program()

    scores = []
    for ex in dev:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        scores.append(result.score)
        status = "OK " if result.correct else "FAIL"
        print(f"[{status}] ({result.details:18s}) Q: {ex.question[:50]}")
        print(f"       Gold: {ex.answer!r:20s}  Pred: {getattr(pred, 'answer', '')!r}")

    avg = sum(scores) / len(scores)
    print("-" * 80)
    print(f"Dev examples : {len(scores)}")
    print(f"Average score: {avg:.3f}")


if __name__ == "__main__":
    main()
