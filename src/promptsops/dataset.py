from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dspy
from datasets import load_dataset


def map_tinyqa_record(row: Mapping[str, Any]) -> dspy.Example:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Tiny QA row is missing a valid 'metadata' object.")

    context = metadata.get("context")
    question = row.get("text")
    answer = row.get("label")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Tiny QA row has an empty or invalid 'metadata.context' value.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Tiny QA row has an empty or invalid 'text' value.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Tiny QA row has an empty or invalid 'label' value.")

    return dspy.Example(context=context, question=question, answer=answer).with_inputs(
        "context", "question"
    )


def load_tinyqa_examples(split_ratio: float = 0.7) -> tuple[list[dspy.Example], list[dspy.Example]]:
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be between 0 and 1, exclusive. Got: {split_ratio}")

    ds = load_dataset("vincentkoc/tiny_qa_benchmark")
    records = list(ds["train"])
    if len(records) < 2:
        raise ValueError("Expected at least 2 records in tiny_qa_benchmark.")

    examples = [map_tinyqa_record(row) for row in records]

    split_idx = int(len(examples) * split_ratio)
    split_idx = max(1, min(len(examples) - 1, split_idx))
    train = examples[:split_idx]
    dev = examples[split_idx:]
    return train, dev
