from types import SimpleNamespace

import pytest

import promptsops.eval_runner as eval_runner


class _Program:
    def __call__(self, *, context: str, question: str):
        del context, question
        return SimpleNamespace(answer="stub")


def test_evaluate_program_returns_average_score(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_runner,
        "load_runtime_config",
        lambda: SimpleNamespace(generator_model="llama3.2:3b"),
    )
    monkeypatch.setattr(eval_runner, "assert_ollama_ready", lambda required_models: None)
    monkeypatch.setattr(eval_runner, "configure_lm", lambda model_name: None)

    dev_examples = [
        SimpleNamespace(context="c1", question="q1", answer="a1"),
        SimpleNamespace(context="c2", question="q2", answer="a2"),
    ]
    monkeypatch.setattr(eval_runner, "load_tinyqa_examples", lambda: ([], dev_examples))
    monkeypatch.setattr(eval_runner, "TinyQAProgram", _Program)

    scores = iter([0.4, 0.8])
    monkeypatch.setattr(
        eval_runner,
        "deterministic_metric",
        lambda example, prediction: SimpleNamespace(score=next(scores)),
    )

    assert eval_runner.evaluate_program() == pytest.approx(0.6)
