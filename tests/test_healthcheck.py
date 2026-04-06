from promptsops.healthcheck import ollama_healthcheck


def test_ollama_is_running():
    assert ollama_healthcheck() is True
