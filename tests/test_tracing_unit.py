from promptsops.tracing import is_tracing_enabled


def test_is_tracing_enabled_accepts_truthy_values(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_TRACING", "true")
    assert is_tracing_enabled() is True

    monkeypatch.setenv("ENABLE_TRACING", "1")
    assert is_tracing_enabled() is True


def test_is_tracing_enabled_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_TRACING", raising=False)
    assert is_tracing_enabled() is False
