"""
Base models for conductor
"""
from langchain.pydantic_v1 import BaseModel, Field


class BaseConductorToolInput(BaseModel):
    job_id: str = Field("The job id of the job that is being run, is a UUID4")
