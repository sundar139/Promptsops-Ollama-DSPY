import dspy

from .signatures import TinyQASignature


class TinyQAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(TinyQASignature)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.predict(context=context, question=question)
