"""
Base models for conductor
"""
from langchain.pydantic_v1 import BaseModel as LangchainBaseModel
from langchain.pydantic_v1 import Field as LangchainField
from pydantic import BaseModel


class BaseConductorToolInput(LangchainBaseModel):
    job_id: str = LangchainField("The job id of the job that is being run, is a UUID4")


class InternalKnowledgeChat(BaseModel):
    id: str
    message: str
    author: str
    created_at: str
    source: str
    channel: str


class ConductorJobCustomerInput(BaseModel):
    job_name: str
    job_id: str
    geography: str = None
    titles: list[str] = None
    industries: list[str] = None


class ConductorJobCustomerResponse(BaseModel):
    input: ConductorJobCustomerInput
    agent_query: str = None
    response: list = None
