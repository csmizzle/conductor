"""
Base models for conductor
"""
from langchain.pydantic_v1 import BaseModel as LanhchainBaseModel
from langchain.pydantic_v1 import Field as LangchainField
from pydantic import BaseModel


class BaseConductorToolInput(LanhchainBaseModel):
    job_id: str = LangchainField("The job id of the job that is being run, is a UUID4")


class InternalKnowledgeChat(BaseModel):
    id: str
    message: str
    author: str
    created_at: str
    source: str
    channel: str
