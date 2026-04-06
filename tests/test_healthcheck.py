import pytest

from promptsops.healthcheck import assert_ollama_ready


@pytest.mark.integration
def test_ollama_is_running():
    status = assert_ollama_ready()
    assert status.reachable is True
    assert not status.missing_models
