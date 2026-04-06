import pytest
from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm
from promptsops.dataset import load_tinyqa_examples
from promptsops.metrics import llm_as_judge_metric


@pytest.mark.slow
def test_tinyqa_llm_judge_sample():
    configure_lm()
    _, dev = load_tinyqa_examples()
    program = load_compiled_program()

    subset = dev[:10]
    correct = 0
    for ex in subset:
        pred = program(context=ex.context, question=ex.question)
        result = llm_as_judge_metric(ex, pred)
        if result.correct:
            correct += 1

    assert correct >= 5, (
        f"Judge scored only {correct}/10 correct. "
        "Threshold reflects llama3.2:1b judge capacity. "
        "Upgrade judge model to raise this threshold."
    )
