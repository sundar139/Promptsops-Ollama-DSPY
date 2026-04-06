from pathlib import Path

import pytest

from promptsops.artifacts import load_compiled_program, resolve_artifact_path


def test_resolve_artifact_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        resolve_artifact_path(tmp_path)


def test_load_compiled_program_missing_file_message(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Run scripts/optimize.py"):
        load_compiled_program(missing)
