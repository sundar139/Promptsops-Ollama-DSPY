from __future__ import annotations

import os
import dspy

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama3.2:3b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.2:1b")
DEFAULT_TEMPERATURE = float(os.getenv("DSPY_TEMPERATURE", "0.2"))


def configure_lm(model_name: str | None = None, temperature: float | None = None) -> dspy.LM:
    lm = dspy.LM(
        model=f"ollama_chat/{model_name or GENERATOR_MODEL}",
        api_base=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
    )
    dspy.configure(lm=lm)
    return lm
