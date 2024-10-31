from pydantic import BaseModel, Field
from crewai import Agent, Task
from abc import ABC, abstractmethod


class Team(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]


class SearchAgent(BaseModel):
    title: str
    questions: list[str]


class SearchTeam(BaseModel):
    title: str
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
    def _build_expected_output(self, task_description: str) -> str:
        pass

    @abstractmethod
    def build(self) -> Task:
        pass


class TeamFactory(ABC):
    @abstractmethod
    def build(self) -> Team:
        pass


class CitedAnswer(BaseModel):
    answer: str = Field(description="The answer for the question")
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: float = Field(
        ge=0, le=1, description="The faithfulness of the answer"
    )
    factual_correctness: float = Field(
        ge=0, le=1, description="The factual correctness of the answer"
    )
    confidence: float = Field(ge=0, le=1, description="The confidence of the answer")
