## Summary

- What changed?
- Why was this needed?

## Validation

Paste exact commands run locally and key outputs.

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts tests
uv run pytest -q -m "not integration and not slow" --cov=promptsops --cov-report=term-missing --cov-report=xml
```

## Scope checks

- [ ] Change is focused and avoids unrelated refactors.
- [ ] README/docs updated if behavior changed.
- [ ] Any new runtime assumptions are documented.
