from __future__ import annotations

import re
from dataclasses import dataclass

import dspy


@dataclass
class EvalResult:
    correct: bool
    score: float
    details: str


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def deterministic_metric(example: dspy.Example, pred: dspy.Prediction) -> EvalResult:
    gold = normalize(example.answer)
    got = normalize(getattr(pred, "answer", ""))

    if not got:
        return EvalResult(correct=False, score=0.0, details="empty")

    if got == gold:
        return EvalResult(correct=True, score=1.0, details="exact_match")

    if gold in got or got in gold:
        return EvalResult(correct=True, score=0.7, details="substring_overlap")

    return EvalResult(correct=False, score=0.0, details="mismatch")


class TinyQAJudge(dspy.Module):
    class JudgeSignature(dspy.Signature):
        """Judge whether the predicted answer is correct given the context and gold answer."""

        context: str = dspy.InputField()
        question: str = dspy.InputField()
        gold_answer: str = dspy.InputField()
        predicted_answer: str = dspy.InputField()
        is_correct: bool = dspy.OutputField()

    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict(self.JudgeSignature)

    def forward(
        self, context: str, question: str, gold_answer: str, predicted_answer: str
    ) -> dspy.Prediction:
        return self.judge(
            context=context,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )


def llm_as_judge_metric(example: dspy.Example, pred: dspy.Prediction) -> EvalResult:
    from promptsops.config import configure_lm, load_runtime_config

    config = load_runtime_config()
    configure_lm(model_name=config.judge_model, temperature=0.0)
    judge = TinyQAJudge()

    result = judge(
        context=example.context,
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=getattr(pred, "answer", ""),
    )

    is_correct = bool(getattr(result, "is_correct", False))
    return EvalResult(correct=is_correct, score=1.0 if is_correct else 0.0, details="llm_judge")
