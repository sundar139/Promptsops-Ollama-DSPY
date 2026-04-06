import pytest

import promptsops.config as config_module


def test_load_runtime_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("DSPY_TEMPERATURE", raising=False)

    config = config_module.load_runtime_config()

    assert config.ollama_base_url == "http://localhost:11434"
    assert config.generator_model == "llama3.2:3b"
    assert config.judge_model == "llama3.2:1b"
    assert config.dspy_temperature == 0.2


def test_load_runtime_config_invalid_temperature(monkeypatch) -> None:
    monkeypatch.setenv("DSPY_TEMPERATURE", "not-a-float")

    with pytest.raises(ValueError, match="DSPY_TEMPERATURE"):
        config_module.load_runtime_config()


def test_configure_lm_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="Model name is empty"):
        config_module.configure_lm(model_name="   ")


def test_configure_lm_builds_expected_lm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(config_module.dspy, "LM", FakeLM)
    monkeypatch.setattr(
        config_module.dspy, "configure", lambda lm: captured.setdefault("configured", lm)
    )

    lm = config_module.configure_lm(model_name="llama3.2:3b", temperature=0.0)

    assert isinstance(lm, FakeLM)
    assert captured["model"] == "ollama_chat/llama3.2:3b"
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["api_key"] == "ollama"
    assert captured["temperature"] == 0.0
    assert captured["configured"] is lm
