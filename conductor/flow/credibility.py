"""
Source credibility module using dspy
"""
import dspy
from pydantic import BaseModel, Field
from enum import Enum

llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)


class SourceCredibilityEnum(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Source(BaseModel):
    source: str = Field(description="The source of the information")


class SourceCredibility(BaseModel):
    source: str = Field(description="The source of the information")
    credibility: SourceCredibilityEnum = Field(
        description="The credibility of the source"
    )

    class Config:
        use_enum_values = True


class SourceCredibilityAnalysisSignature(dspy.Signature):
    source: Source = dspy.InputField()
    credibility: SourceCredibility = dspy.OutputField()


source_analysis = dspy.ChainOfThought(SourceCredibilityAnalysisSignature)


def get_source_credibility(source: str) -> dspy.Prediction:
    return source_analysis(source=Source(source=source))
