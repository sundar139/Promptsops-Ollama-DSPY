from pathlib import Path

import pytest

from promptsops.artifacts import ARTIFACT_PATH


@pytest.mark.integration
def test_compiled_artifact_exists():
    assert Path(ARTIFACT_PATH).exists()
