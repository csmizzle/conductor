"""
Source credibility module using dspy
"""
import dspy
from pydantic import BaseModel, Field
from enum import Enum

llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)


class SourceCredibility(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Source(BaseModel):
    source: str = Field(description="The source of the information")


class SourceCredibility(BaseModel):
    source: str = Field(description="The source of the information")
    credibility: SourceCredibility = Field(description="The credibility of the source")


class SourceCredibilityAnalysisSignature(dspy.Signature):
    source: Source = dspy.InputField()
    credibility: SourceCredibility = dspy.OutputField()


source_analysis = dspy.TypedChainOfThought(SourceCredibilityAnalysisSignature)


def get_source_credibility(source: str):
    return source_analysis(source=Source(source=source))
