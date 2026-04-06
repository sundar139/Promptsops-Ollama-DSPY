import pytest

from promptsops.artifacts import load_compiled_program
from promptsops.config import configure_lm, load_runtime_config
from promptsops.dataset import load_tinyqa_examples
from promptsops.healthcheck import assert_ollama_ready
from promptsops.metrics import llm_as_judge_metric


@pytest.mark.slow
@pytest.mark.integration
def test_tinyqa_llm_judge_sample():
    config = load_runtime_config()
    assert_ollama_ready(required_models=(config.generator_model, config.judge_model))
    configure_lm(model_name=config.generator_model)
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
