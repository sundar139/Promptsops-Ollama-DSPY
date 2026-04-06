import json
from pathlib import Path

from promptsops.results import BenchmarkResult, save_benchmark_result


def test_save_benchmark_result_writes_run_and_latest_files(tmp_path: Path) -> None:
    result = BenchmarkResult(
        model_name="llama3.2:3b",
        train_size=36,
        dev_size=16,
        deterministic_score=0.731,
    )

    run_path, latest_path = save_benchmark_result(result, output_dir=tmp_path)

    assert run_path.exists()
    assert latest_path.exists()
    assert latest_path.name == "latest.json"

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "llama3.2:3b"
    assert payload["train_size"] == 36
    assert payload["dev_size"] == 16
    assert payload["deterministic_score"] == 0.731
    assert payload["timestamp_utc"]

    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest_payload == payload
