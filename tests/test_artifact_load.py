from pathlib import Path
from promptsops.artifacts import ARTIFACT_PATH


def test_compiled_artifact_exists():
    assert Path(ARTIFACT_PATH).exists()
