from pydantic import BaseModel
from crewai import Agent, Task


class Team(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]
