from pydantic import BaseModel
from crewai import Agent, Task
from abc import ABC, abstractmethod


class Team(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]


class AgentFactory(ABC):
    @abstractmethod
    def build(self) -> Agent:
        pass


class TaskFactory(ABC):
    @abstractmethod
    def build(self) -> Task:
        pass


class TeamFactory(ABC):
    @abstractmethod
    def build(self) -> Team:
        pass
