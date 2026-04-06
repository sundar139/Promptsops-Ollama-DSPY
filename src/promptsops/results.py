from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("artifacts/benchmark_results")
LATEST_RESULT_PATH = RESULTS_DIR / "latest.json"


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    train_size: int
    dev_size: int
    deterministic_score: float
    timestamp_utc: str = ""


def _timestamp_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _result_filename(timestamp_utc: str) -> str:
    safe = timestamp_utc.replace("+00:00", "Z").replace(":", "").replace("-", "")
    return f"run_{safe}.json"


def save_benchmark_result(
    result: BenchmarkResult,
    output_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_dir) if output_dir is not None else RESULTS_DIR
    destination.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    if not payload["timestamp_utc"]:
        payload["timestamp_utc"] = _timestamp_utc()

    run_path = destination / _result_filename(payload["timestamp_utc"])
    run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    latest_path = destination / LATEST_RESULT_PATH.name
    latest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return run_path, latest_path
