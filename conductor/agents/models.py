"""
Model for agentic work in Evrim
"""
from pydantic import BaseModel, Field


class Score(BaseModel):
    relevance: bool = Field(description="Is the document relevant to the question?")
