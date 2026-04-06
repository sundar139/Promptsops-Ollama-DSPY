import dspy


class TinyQASignature(dspy.Signature):
    """Answer the question concisely and factually using the given context."""

    context: str = dspy.InputField(desc="Supporting context sentence")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Short factual answer")
