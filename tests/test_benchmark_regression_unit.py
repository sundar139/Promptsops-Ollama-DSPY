import json
from pathlib import Path

import pytest

from promptsops.benchmark_regression import compare_latest_to_previous, evaluate_regression_gate


def _write_result(path: Path, score: float, timestamp_utc: str) -> None:
    payload = {
        "model_name": "llama3.2:3b",
        "train_size": 36,
        "dev_size": 16,
        "deterministic_score": score,
        "timestamp_utc": timestamp_utc,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_latest_to_previous_finds_baseline(tmp_path: Path) -> None:
    _write_result(tmp_path / "run_20260406T040000Z.json", 0.70, "2026-04-06T04:00:00Z")
    _write_result(tmp_path / "run_20260406T050000Z.json", 0.73, "2026-04-06T05:00:00Z")
    _write_result(tmp_path / "latest.json", 0.73, "2026-04-06T05:00:00Z")

    comparison = compare_latest_to_previous(tmp_path)

    assert comparison.latest.score == pytest.approx(0.73)
    assert comparison.baseline is not None
    assert comparison.baseline.score == pytest.approx(0.70)
    assert comparison.delta == pytest.approx(0.03)


def test_compare_latest_to_previous_without_baseline(tmp_path: Path) -> None:
    _write_result(tmp_path / "run_20260406T050000Z.json", 0.73, "2026-04-06T05:00:00Z")
    _write_result(tmp_path / "latest.json", 0.73, "2026-04-06T05:00:00Z")

    comparison = compare_latest_to_previous(tmp_path)
    ok, message = evaluate_regression_gate(comparison, max_regression=0.03)

    assert comparison.baseline is None
    assert ok is True
    assert "skipping" in message.lower()


def test_evaluate_regression_gate_fails_on_large_drop(tmp_path: Path) -> None:
    _write_result(tmp_path / "run_20260406T040000Z.json", 0.80, "2026-04-06T04:00:00Z")
    _write_result(tmp_path / "run_20260406T050000Z.json", 0.72, "2026-04-06T05:00:00Z")
    _write_result(tmp_path / "latest.json", 0.72, "2026-04-06T05:00:00Z")

    comparison = compare_latest_to_previous(tmp_path)
    ok, message = evaluate_regression_gate(comparison, max_regression=0.03)

    assert ok is False
    assert "failed" in message.lower()


def test_compare_latest_to_previous_requires_latest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="latest benchmark"):
        compare_latest_to_previous(tmp_path)
