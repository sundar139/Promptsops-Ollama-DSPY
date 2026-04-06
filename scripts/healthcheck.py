from __future__ import annotations

from promptsops.config import get_required_models, load_runtime_config
from promptsops.healthcheck import assert_ollama_ready


def main() -> None:
    config = load_runtime_config()
    required_models = get_required_models(config)
    status = assert_ollama_ready(required_models=required_models)

    print(status.message)
    print(f"Ollama base URL: {config.ollama_base_url}")
    print(f"Required models: {', '.join(required_models)}")


if __name__ == "__main__":
    main()
