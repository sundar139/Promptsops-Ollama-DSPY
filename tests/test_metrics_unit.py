from promptsops.metrics import EvalResult, deterministic_metric, normalize


class _Pred:
    def __init__(self, answer: str):
        self.answer = answer


class _Example:
    def __init__(self, answer: str):
        self.answer = answer


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  The   QUICK\nBrown\tFox  ") == "the quick brown fox"


def test_deterministic_metric_exact_match() -> None:
    result = deterministic_metric(_Example("Yen"), _Pred(" yen "))
    assert result == EvalResult(correct=True, score=1.0, details="exact_match")


def test_deterministic_metric_substring_match() -> None:
    result = deterministic_metric(_Example("Yen"), _Pred("Japanese yen"))
    assert result.correct is True
    assert result.score == 0.7
    assert result.details == "substring_overlap"


def test_deterministic_metric_empty_prediction() -> None:
    result = deterministic_metric(_Example("Yen"), _Pred(""))
    assert result.correct is False
    assert result.score == 0.0
    assert result.details == "empty"
