from __future__ import annotations

from pathlib import Path

import dspy

from .program import TinyQAProgram

ARTIFACT_PATH = Path("artifacts/compiled_program.json")


def resolve_artifact_path(path: str | Path | None = None) -> Path:
    resolved = Path(path) if path is not None else ARTIFACT_PATH
    if resolved.is_dir():
        raise IsADirectoryError(f"Expected a file path but got a directory: {resolved}")
    return resolved


def load_compiled_program(path: str | Path | None = None) -> dspy.Module:
    resolved = resolve_artifact_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Compiled artifact not found at {resolved}. Run scripts/optimize.py to generate it."
        )

    base = TinyQAProgram()
    base.load(str(resolved))
    return base
