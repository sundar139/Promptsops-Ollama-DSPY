# Architecture

A compact view of the local PromptsOps pipeline.

```mermaid
flowchart LR
    DS[Tiny QA Dataset Loader\nsrc/promptsops/dataset.py] --> PRG[DSPy Program\nsrc/promptsops/program.py]
    OLL[Ollama Runtime\n(generator/judge models)] --> PRG
    PRG --> OPT[Optimizer\nsrc/promptsops/optimizer.py]
    OPT --> ART[Compiled Artifact\nartifacts/compiled_program.json]
    ART --> EVAL[Evaluation + Tests\nscripts/run_eval.py + pytest]
    PRG -. optional .-> TRC[OTel Tracing\nsrc/promptsops/tracing.py]
    EVAL --> BENCH[Benchmark JSON\nartifacts/benchmark_results/]
```

## Components

- Dataset loader: maps Tiny QA rows into typed DSPy examples.
- DSPy program: inference module over context + question.
- Ollama runtime: local LLM serving for generator/judge models.
- Optimizer: compiles demonstrations into a reusable artifact.
- Evaluation/tests: deterministic scoring plus optional LLM-as-judge.
- Tracing: opt-in OpenTelemetry export for runtime inspection.
