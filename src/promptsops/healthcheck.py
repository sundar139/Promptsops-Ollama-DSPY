from __future__ import annotations

import httpx
from promptsops.config import OLLAMA_BASE_URL


def ollama_healthcheck(timeout: float = 5.0) -> bool:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        return True
    except Exception:
        return False
