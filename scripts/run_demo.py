from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.program import TinyQAProgram


def main() -> None:
    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model,))
    configure_lm(model_name=config.generator_model)
    _, dev = load_tinyqa_examples()
    program = TinyQAProgram()

    for ex in dev[:5]:
        pred = program(context=ex.context, question=ex.question)
        print("-" * 80)
        print(f"Q: {ex.question}")
        print(f"Gold: {ex.answer}")
        print(f"Pred: {getattr(pred, 'answer', '')}")


if __name__ == "__main__":
    main()
