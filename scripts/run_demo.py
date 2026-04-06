from promptsops.config import configure_lm
from promptsops.dataset import load_tinyqa_examples
from promptsops.program import TinyQAProgram


def main() -> None:
    configure_lm()
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
