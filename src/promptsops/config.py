from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import dspy

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_GENERATOR_MODEL = "llama3.2:3b"
DEFAULT_JUDGE_MODEL = "llama3.2:1b"
DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_base_url: str
    generator_model: str
    judge_model: str
    dspy_temperature: float


def _read_non_empty_env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} is empty. Please set a non-empty value.")
    return value


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float. Got: {raw!r}") from exc
    return value


def load_runtime_config() -> RuntimeConfig:
    base_url = _read_non_empty_env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "OLLAMA_BASE_URL must be an absolute http(s) URL, "
            f"for example {DEFAULT_OLLAMA_BASE_URL!r}. Got: {base_url!r}"
        )

    generator_model = _read_non_empty_env("GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL)
    judge_model = _read_non_empty_env("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    dspy_temperature = _read_float_env("DSPY_TEMPERATURE", DEFAULT_TEMPERATURE)
    if not 0.0 <= dspy_temperature <= 2.0:
        raise ValueError(
            f"DSPY_TEMPERATURE must be between 0.0 and 2.0 inclusive. Got: {dspy_temperature}"
        )

    return RuntimeConfig(
        ollama_base_url=base_url,
        generator_model=generator_model,
        judge_model=judge_model,
        dspy_temperature=dspy_temperature,
    )


def get_required_models(config: RuntimeConfig | None = None) -> tuple[str, ...]:
    cfg = config or load_runtime_config()
    if cfg.generator_model == cfg.judge_model:
        return (cfg.generator_model,)
    return (cfg.generator_model, cfg.judge_model)


def configure_lm(model_name: str | None = None, temperature: float | None = None) -> dspy.LM:
    config = load_runtime_config()
    selected_model = model_name.strip() if model_name is not None else config.generator_model
    if not selected_model:
        raise ValueError("Model name is empty. Set GENERATOR_MODEL or pass model_name explicitly.")

    selected_temperature = config.dspy_temperature if temperature is None else temperature
    if not 0.0 <= selected_temperature <= 2.0:
        raise ValueError(
            f"temperature must be between 0.0 and 2.0 inclusive. Got: {selected_temperature}"
        )

    lm = dspy.LM(
        model=f"ollama_chat/{selected_model}",
        api_base=config.ollama_base_url,
        api_key="ollama",
        temperature=selected_temperature,
    )
    dspy.configure(lm=lm)
    return lm
