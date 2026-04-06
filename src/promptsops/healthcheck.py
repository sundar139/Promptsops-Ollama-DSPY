from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from promptsops.config import get_required_models, load_runtime_config


@dataclass(frozen=True)
class OllamaHealthStatus:
    reachable: bool
    available_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    message: str


def _extract_model_names(payload: dict[str, Any]) -> tuple[str, ...]:
    models = payload.get("models", [])
    if not isinstance(models, list):
        return tuple()

    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    return tuple(names)


def check_ollama(
    required_models: tuple[str, ...] | None = None, timeout: float = 5.0
) -> OllamaHealthStatus:
    config = load_runtime_config()
    expected_models = required_models or get_required_models(config)

    try:
        response = httpx.get(f"{config.ollama_base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        available_models = _extract_model_names(response.json())
    except httpx.HTTPError as exc:
        return OllamaHealthStatus(
            reachable=False,
            available_models=tuple(),
            missing_models=expected_models,
            message=(
                "Ollama is unavailable. Ensure the server is running and reachable at "
                f"{config.ollama_base_url!r}. Original error: {exc}"
            ),
        )

    missing_models = tuple(model for model in expected_models if model not in available_models)
    if missing_models:
        missing = ", ".join(missing_models)
        return OllamaHealthStatus(
            reachable=True,
            available_models=available_models,
            missing_models=missing_models,
            message=(
                "Ollama is running but required models are missing: "
                f"{missing}. Pull them with: "
                + " ".join(f"ollama pull {model}" for model in missing_models)
            ),
        )

    return OllamaHealthStatus(
        reachable=True,
        available_models=available_models,
        missing_models=tuple(),
        message="Ollama is reachable and required models are available.",
    )


def assert_ollama_ready(
    required_models: tuple[str, ...] | None = None, timeout: float = 5.0
) -> OllamaHealthStatus:
    status = check_ollama(required_models=required_models, timeout=timeout)
    if not status.reachable or status.missing_models:
        raise RuntimeError(status.message)
    return status


def ollama_healthcheck(timeout: float = 5.0) -> bool:
    status = check_ollama(timeout=timeout)
    return status.reachable and not status.missing_models
