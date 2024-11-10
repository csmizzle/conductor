from pydantic import BaseModel, Field
from crewai import Agent, Task
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union


class Team(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]


class SearchAgent(BaseModel):
    title: str
    questions: list[str]


class SearchTeam(BaseModel):
    title: str
    perspective: str
    agents: list[SearchAgent]


class AgentFactory(ABC):
    @abstractmethod
    def _build_backstory(self) -> str:
        pass

    @abstractmethod
    def _build_goal(self) -> str:
        pass

    @abstractmethod
    def build(self) -> Agent:
        pass


class TaskFactory(ABC):
    @abstractmethod
    def _build_description(self) -> str:
        pass

    @abstractmethod
    def build(self) -> Task:
        pass


class TeamFactory(ABC):
    @abstractmethod
    def build(self) -> Team:
        pass


class NotAvailable(Enum):
    """Not available"""

    NOT_AVAILABLE = "NOT_AVAILABLE"


class CitedAnswer(BaseModel):
    answer: Union[str, NotAvailable] = Field(description="The answer for the question")
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the answer")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the answer"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the answer")

    class Config:
        use_enum_values = True


class CitedValue(BaseModel):
    """Best value for a question"""

    value: Union[str, bool, NotAvailable] = Field(
        description="The value for the question"
    )
    citations: list[str] = Field(description="The URLs used in the value")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the value")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the value"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the value")

    class Config:
        use_enum_values = True
