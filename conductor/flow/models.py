from pydantic import BaseModel
from crewai import Agent, Task
from abc import ABC, abstractmethod


class Team(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]


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
