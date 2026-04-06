"""Debug failed QA examples with optional Phoenix trace export."""

from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import deterministic_metric, normalize
from promptsops.tracing import is_tracing_enabled, start_tracing


def main() -> None:
    if is_tracing_enabled():
        start_tracing()
        print("Tracing enabled — exporting spans to Phoenix")

    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    _, dev = load_tinyqa_examples()
    program = load_compiled_program()

    failures = []
    for ex in dev:
        pred = program(context=ex.context, question=ex.question)
        result = deterministic_metric(ex, pred)
        if not result.correct:
            failures.append((ex, pred, result))

    total = len(dev)
    passed = total - len(failures)
    print(f"\nResults: {passed}/{total} passed, {len(failures)}/{total} failed\n")

    if not failures:
        print("No failures — dev set fully correct.")
        return

    print("=" * 80)
    print(f"FAILURES ({len(failures)} examples)")
    print("=" * 80)

    for i, (ex, pred, result) in enumerate(failures, 1):
        raw_pred = getattr(pred, "answer", "")
        print(f"\n[{i}/{len(failures)}]  detail={result.details}")
        print(f"  Context : {ex.context}")
        print(f"  Question: {ex.question}")
        print(f"  Gold    : {ex.answer!r:30s}  normalized: {normalize(ex.answer)!r}")
        print(f"  Pred    : {raw_pred!r:30s}  normalized: {normalize(raw_pred)!r}")


if __name__ == "__main__":
    main()
