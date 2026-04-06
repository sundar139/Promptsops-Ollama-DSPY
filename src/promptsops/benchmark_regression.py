from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from promptsops.results import RESULTS_DIR


@dataclass(frozen=True)
class BenchmarkSnapshot:
    path: Path
    score: float
    timestamp_utc: str


@dataclass(frozen=True)
class BenchmarkComparison:
    latest: BenchmarkSnapshot
    baseline: BenchmarkSnapshot | None

    @property
    def delta(self) -> float | None:
        if self.baseline is None:
            return None
        return self.latest.score - self.baseline.score


def _load_snapshot(path: Path) -> BenchmarkSnapshot:
    payload = json.loads(path.read_text(encoding="utf-8"))
    score = payload.get("deterministic_score")
    if not isinstance(score, int | float):
        raise ValueError(f"Benchmark file {path} is missing numeric deterministic_score")

    timestamp_utc = payload.get("timestamp_utc")
    if not isinstance(timestamp_utc, str) or not timestamp_utc:
        raise ValueError(f"Benchmark file {path} is missing non-empty timestamp_utc")

    return BenchmarkSnapshot(path=path, score=float(score), timestamp_utc=timestamp_utc)


def _find_previous_run(results_dir: Path, latest: BenchmarkSnapshot) -> BenchmarkSnapshot | None:
    run_files = sorted(results_dir.glob("run_*.json"))
    for run_file in reversed(run_files):
        candidate = _load_snapshot(run_file)
        if candidate.timestamp_utc != latest.timestamp_utc:
            return candidate
    return None


def compare_latest_to_previous(results_dir: Path | str = RESULTS_DIR) -> BenchmarkComparison:
    directory = Path(results_dir)
    latest_path = directory / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest benchmark file found at {latest_path}. Run scripts/run_eval.py first."
        )

    latest = _load_snapshot(latest_path)
    baseline = _find_previous_run(directory, latest)
    return BenchmarkComparison(latest=latest, baseline=baseline)


def evaluate_regression_gate(
    comparison: BenchmarkComparison,
    max_regression: float | None,
) -> tuple[bool, str]:
    if comparison.baseline is None:
        return True, "No prior benchmark run found; skipping regression gate."

    delta = comparison.delta
    assert delta is not None

    if max_regression is None:
        return True, (
            f"Latest score: {comparison.latest.score:.3f} | "
            f"Baseline: {comparison.baseline.score:.3f} | "
            f"Delta: {delta:+.3f}"
        )

    if delta < -max_regression:
        return False, (
            f"Regression gate failed. Latest score: {comparison.latest.score:.3f} | "
            f"Baseline: {comparison.baseline.score:.3f} | "
            f"Delta: {delta:+.3f} exceeds allowed drop {-max_regression:.3f}."
        )

    return True, (
        f"Regression gate passed. Latest score: {comparison.latest.score:.3f} | "
        f"Baseline: {comparison.baseline.score:.3f} | "
        f"Delta: {delta:+.3f}"
    )
