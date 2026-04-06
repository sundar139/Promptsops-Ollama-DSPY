from __future__ import annotations

from pathlib import Path
import dspy

from .program import TinyQAProgram

ARTIFACT_PATH = Path("artifacts/compiled_program.json")


def load_compiled_program(path: str | None = None) -> dspy.Module:
    resolved = Path(path) if path else ARTIFACT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Compiled artifact not found: {resolved}")

    base = TinyQAProgram()
    base.load(str(resolved))
    return base
