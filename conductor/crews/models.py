from pydantic.v1 import BaseModel, Field


class TaskRun(BaseModel):
    name: str = Field(description="Name of task")
    agent_role: str = Field(description="The name of the agent that ran the task")
    description: str = Field(description="The description of the task that was run")
    result: str = Field(description="The result of the task run")


class CrewRun(BaseModel):
    tasks: list[TaskRun] = Field(
        description="List of tasks that were completed by the crew"
    )
    result: str = Field(description="The result of the crew run")
