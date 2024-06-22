from pydantic import BaseModel, Field
from crewai.task import TaskOutput


class CrewRun(BaseModel):
    task_outputs: list[TaskOutput] = Field(
        description="List of tasks that were completed by the crew"
    )
    result: str = Field(description="The result of the crew run")
