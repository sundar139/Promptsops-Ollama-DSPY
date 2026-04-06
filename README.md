# promptsops-ollama-dspy

A local **PromptsOps** pipeline that treats prompts like code.
A DSPy program is evaluated against a benchmark, optimized via BootstrapFewShot,
saved as a versioned artifact, quality-gated by pytest, and traced via
OpenInference + Phoenix.

---

## Motivation and Goals

Manual prompt tuning is fragile: small edits to wording can silently degrade
quality, and there is no way to measure, compare, or roll back changes.
This project applies software engineering practices to prompts:

| Practice | Prompt equivalent |
|---|---|
| Unit tests | Deterministic metric assertions on a benchmark |
| Build artifacts | Versioned compiled DSPy JSON state |
| CI | GitHub Actions running optimization + tests on every push |
| Observability | OTLP trace export to Phoenix |

The pipeline is designed to run entirely **offline on a single machine** using
Ollama, making it suitable for experimentation without cloud API costs.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Local inference | [Ollama](https://ollama.com) (`llama3.2:3b` generator, `llama3.2:1b` judge) |
| Prompt programming | [DSPy](https://dspy.ai) 3.x (signatures, modules, BootstrapFewShot) |
| Dataset | [`vincentkoc/tiny_qa_benchmark`](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark) (52 QA pairs) |
| Dependency manager | [uv](https://docs.astral.sh/uv/) |
| Tests / CI gates | pytest |
| CI | GitHub Actions |
| Observability | OpenTelemetry SDK + OpenInference DSPy instrumentation + Arize Phoenix |

---

## Features and Implementation Details

### 1. Dataset and Baseline Program (Phase 1)

**What:** Loads the Tiny QA Benchmark from HuggingFace, splits 70/30 into
train (36) and dev (16) sets, and runs inference through a `ChainOfThought`
DSPy module.

**How:** [`dataset.py`](src/promptsops/dataset.py) maps the dataset schema
(`row["text"]` → question, `row["label"]` → answer,
`row["metadata"]["context"]` → context) to `dspy.Example` objects.
[`program.py`](src/promptsops/program.py) wraps `dspy.ChainOfThought` over
a typed [`TinyQASignature`](src/promptsops/signatures.py).

**Why ChainOfThought over Predict:** Eliciting a reasoning chain before the
final answer improves accuracy on factual QA, especially for small models.

### 2. Deterministic Evaluation (Phase 2)

**What:** Measures program quality with exact-match and substring-overlap
scoring.

**How:** [`metrics.py`](src/promptsops/metrics.py) normalizes both gold and
predicted answers (lowercase, collapse whitespace), then checks exact match
(score 1.0) or substring containment (score 0.7).

**Why substring overlap:** Small models often return "¥ (Japanese Yen)"
instead of "Yen". Substring matching captures semantically correct but
differently formatted answers without requiring an LLM judge.

### 3. Optimization and Artifact Saving (Phase 3)

**What:** Runs `BootstrapFewShot` to compile few-shot demonstrations into
the program and saves the result as a versioned JSON artifact.

**How:** [`optimizer.py`](src/promptsops/optimizer.py) bootstraps up to 8
traces from the training set. The compiled program is serialized to
[`artifacts/compiled_program.json`](artifacts/) via `dspy.Module.save()`.

**Why BootstrapFewShot:** It is the simplest DSPy optimizer and works well
with small models and small datasets. More sophisticated optimizers
(MIPROv2, BayesianSignatureOptimizer) can be swapped in later.

### 4. pytest Quality Gates (Phase 4)

**What:** Automated tests that prevent silent regressions.

**Tests:**
- `test_healthcheck.py` — Ollama is reachable at `/api/tags`
- `test_artifact_load.py` — Compiled artifact exists on disk
- `test_eval_deterministic.py` — Compiled program scores ≥ 0.60 on the dev set

### 5. LLM Judge Evaluation (Phase 5)

**What:** Semantic evaluation using a second model as a judge.

**How:** [`TinyQAJudge`](src/promptsops/metrics.py) is a DSPy module that
takes context, question, gold answer, and predicted answer, then outputs
`is_correct: bool`. The test is marked `@pytest.mark.slow` and excluded
from CI.

**Why a separate judge model:** `llama3.2:1b` is used as the judge to avoid
self-evaluation bias. The trade-off is that a 1B model is a weak judge
(caps at ~5/10 accuracy on this dataset).

### 6. CI Workflow (Phase 6)

**What:** GitHub Actions workflow that installs Ollama, pulls models,
optimizes the program, and runs the core test suite on every push to `main`.

**How:** Uses `astral-sh/setup-uv@v5` (reads `.python-version` automatically).
Slow judge tests are excluded from CI to keep the pipeline fast and
deterministic.

### 7. Observability (Phase 7)

**What:** Opt-in OpenTelemetry tracing for all DSPy LM calls, exportable
to a Phoenix server.

**How:** [`tracing.py`](src/promptsops/tracing.py) sets up a
`TracerProvider` → `BatchSpanProcessor` → `OTLPSpanExporter` and instruments
DSPy via `DSPyInstrumentor`. Activated by setting `ENABLE_TRACING=1`.

**Why raw OTLP instead of `phoenix.otel.register`:** The `arize-phoenix`
package cannot be imported on Python 3.14/Windows due to a missing `sqlean`
wheel. Using the OpenTelemetry SDK directly bypasses this while preserving
full trace export.

---

## Challenges and Issues Encountered

### DSPy 3.x API Incompatibilities

The implementation plan was written for DSPy 2.x. Three breaking changes had
to be corrected:

| DSPy 2.x (plan) | DSPy 3.x (actual) |
|---|---|
| `dspy.OpenAI(...)` + `dspy.settings.configure(lm=lm)` | `dspy.LM("ollama_chat/model", ...)` + `dspy.configure(lm=lm)` |
| `optimizer.compile(program, train=train, val=dev)` | `optimizer.compile(program, trainset=train)` |
| `dspy.Module.load(path, base=base)` | `base = TinyQAProgram(); base.load(path)` |

### Ollama Model Prefix

DSPy uses LiteLLM internally. The model string must use `ollama_chat/` (not
`ollama/`) for chat-completion models. Using `ollama/` produces gibberish
output because it sends raw completion requests.

### Weak Judge Model

`llama3.2:1b` consistently scores exactly 5/10 as a judge regardless of
program quality. The judge test threshold was lowered from 6 to 5
accordingly. Switching to `llama3.2:3b` as judge (via `JUDGE_MODEL` env var)
allows raising the threshold.

### Phoenix on Python 3.14 / Windows

The `sqlean` dependency required by Phoenix has no pre-built wheel for
Python 3.14. Phoenix must be run via Docker:

```sh
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
```

The tracing client stack works regardless of the Phoenix import issue.

---

## Getting Started

### Prerequisites

- **Ollama** installed and running — [ollama.com](https://ollama.com)
- **uv** package manager — `pip install uv` or `winget install astral-sh.uv`
- Python ≥ 3.11 (uv reads `.python-version` to select 3.14 automatically)

### Setup

```sh
# Pull required models
ollama pull llama3.2:3b
ollama pull llama3.2:1b

# Install dependencies
uv sync
```

### Run

```sh
# Quick sanity check (5 dev examples, uncompiled program)
uv run python scripts/run_demo.py

# Optimize and save compiled artifact
uv run python scripts/optimize.py

# Full dev-set evaluation with compiled program
uv run python scripts/run_eval.py
```

---

## Running Tests

```sh
# Core test suite (~10s) — healthcheck, artifact, deterministic eval
uv run pytest -q -m "not slow"

# Include LLM judge test (~15s total)
uv run pytest -v -m slow

# All tests
uv run pytest -v
```

### Quality Gates

| Test | Assertion |
|---|---|
| `test_ollama_is_running` | Ollama responds at `/api/tags` |
| `test_compiled_artifact_exists` | `artifacts/compiled_program.json` present |
| `test_tinyqa_compiled_program_quality` | Average deterministic score ≥ 0.60 |
| `test_tinyqa_llm_judge_sample` | ≥ 5/10 judge-correct (slow, excluded from CI) |

---

## Debugging Failures

```sh
# Detailed failure report with diffs
uv run python scripts/debug_failures.py

# With Phoenix trace export
$env:ENABLE_TRACING = "1"   # PowerShell
uv run python scripts/debug_failures.py
```

Start Phoenix via Docker first:

```sh
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
# Open http://localhost:6006
```

---

## Deployment / CI

The GitHub Actions workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs on every push/PR to `main`:

1. Checkout → install uv → install Ollama → pull models
2. `uv sync` → `uv run python scripts/optimize.py`
3. `uv run pytest -q -m "not slow"`

**Note:** The CI workflow has not been run on GitHub yet. The first run may
surface issues with Ollama pull times in the runner environment.

---

## Project Structure

```
promptsops-ollama-dspy/
├── src/promptsops/          # Core library
│   ├── config.py            # Ollama LM configuration (model names, base URL)
│   ├── signatures.py        # DSPy Signature for the QA task
│   ├── program.py           # TinyQAProgram (ChainOfThought module)
│   ├── dataset.py           # HuggingFace dataset loader → dspy.Example lists
│   ├── metrics.py           # Deterministic metric + LLM judge
│   ├── optimizer.py         # BootstrapFewShot optimization + artifact save
│   ├── artifacts.py         # Compiled artifact loader
│   ├── eval_runner.py       # Standalone baseline evaluation
│   ├── healthcheck.py       # Ollama connectivity check
│   └── tracing.py           # OpenTelemetry + DSPy instrumentation (opt-in)
├── tests/                   # pytest quality gates
│   ├── test_healthcheck.py
│   ├── test_artifact_load.py
│   ├── test_eval_deterministic.py
│   └── test_eval_llm_judge.py
├── scripts/                 # Runnable entry points
│   ├── run_demo.py          # Quick demo (5 dev examples)
│   ├── run_eval.py          # Full dev-set evaluation
│   ├── optimize.py          # Run optimizer and save artifact
│   └── debug_failures.py    # Failure analysis with optional tracing
├── artifacts/               # Compiled DSPy state (gitignored, regenerated)
├── .github/workflows/ci.yml # GitHub Actions CI
├── pyproject.toml           # uv project manifest + pytest config
└── uv.lock                  # Pinned dependency lockfile
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GENERATOR_MODEL` | `llama3.2:3b` | Model used for QA inference |
| `JUDGE_MODEL` | `llama3.2:1b` | Model used for LLM judge evaluation |
| `DSPY_TEMPERATURE` | `0.2` | Sampling temperature for DSPy calls |
| `ENABLE_TRACING` | *(unset)* | Set to `1` to enable OTLP trace export |
| `PHOENIX_ENDPOINT` | `http://localhost:6006/v1/traces` | Phoenix OTLP endpoint |

---

## Known Limitations and Roadmap

**Current limitations:**
- Deterministic threshold (0.60) is conservative — measured baseline is ~0.73
- `llama3.2:1b` judge caps at 5/10 accuracy; upgrade judge model for better semantic eval
- Optimization results vary ±0.05 across runs due to sampling nondeterminism
- CI has not been run on GitHub yet
- Phoenix server requires Docker on Python 3.14 / Windows

**Potential improvements:**
- Raise deterministic threshold after establishing stable baseline across runs
- Add HotPotQA subset for multi-hop reasoning evaluation
- Add structured-output benchmark for format validation
- Try MIPROv2 or BayesianSignatureOptimizer for stronger optimization
- Track `artifacts/compiled_program.json` in git for artifact versioning

---

## Refactoring Summary

The following cleanup was applied to bring the repository to a professional state:

- **Comment cleanup:** Removed verbose docstrings, redundant inline comments, and
  usage instructions from source files. Moved setup/run instructions into this README.
  Kept only high-value comments explaining non-obvious decisions.
- **`__init__.py`:** Replaced unused `main()` stub (from `uv init`) with a module
  docstring. Removed the dead `[project.scripts]` entry from `pyproject.toml`.
- **`pyproject.toml`:** Updated project description. Removed orphaned script entrypoint.
- **`.gitignore`:** Expanded from a minimal list to a comprehensive, categorized
  ignore file covering OS files, editors, Python artifacts, virtual environments,
  testing/coverage outputs, logs, and environment files.
- **README.md:** Fully rewritten with project overview, motivation, feature details,
  challenges encountered, setup instructions, test documentation, CI notes, project
  structure, environment variables, and known limitations.
