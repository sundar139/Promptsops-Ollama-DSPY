from __future__ import annotations

from datasets import load_dataset
import dspy


def load_tinyqa_examples(split_ratio: float = 0.7) -> tuple[list[dspy.Example], list[dspy.Example]]:
    ds = load_dataset("vincentkoc/tiny_qa_benchmark")
    records = list(ds["train"])

    examples: list[dspy.Example] = []
    for row in records:
        context = row["metadata"]["context"]
        question = row["text"]
        answer = row["label"]
        ex = dspy.Example(context=context, question=question, answer=answer).with_inputs(
            "context", "question"
        )
        examples.append(ex)

    split_idx = int(len(examples) * split_ratio)
    train = examples[:split_idx]
    dev = examples[split_idx:]
    return train, dev
