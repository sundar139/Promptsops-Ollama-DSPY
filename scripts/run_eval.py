from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import deterministic_metric
from promptsops.results import BenchmarkResult, save_benchmark_result
from promptsops.tracing import is_tracing_enabled, start_tracing


def main() -> None:
    if is_tracing_enabled():
        start_tracing()
        print("Tracing enabled — exporting spans to Phoenix")

    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    train, dev = load_tinyqa_examples()
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
    benchmark = BenchmarkResult(
        model_name=config.generator_model,
        train_size=len(train),
        dev_size=len(dev),
        deterministic_score=avg,
    )
    run_result_path, latest_result_path = save_benchmark_result(benchmark)

    print("-" * 80)
    print(f"Dev examples : {len(scores)}")
    print(f"Average score: {avg:.3f}")
    print(f"Saved benchmark result: {run_result_path}")
    print(f"Updated latest benchmark result: {latest_result_path}")


if __name__ == "__main__":
    main()
