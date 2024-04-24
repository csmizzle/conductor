"""
tools for querying knowledge bases
"""
from langchain_core.tools import tool
from conductor.retrievers.pinecone_ import (
    create_gpt4_pinecone_apollo_retriever,
    create_gpt4_pinecone_discord_retriever,
)
from langchain.pydantic_v1 import BaseModel, Field
from conductor.models import BaseConductorToolInput
from conductor.functions.apollo import generate_apollo_person_search_context
from conductor.chains import create_apollo_input
import logging
from langsmith import traceable


logger = logging.getLogger(__name__)


class Query(BaseModel):
    query: str = Field("A natural language query")


class QueryWithJobId(Query):
    job_id: str = Field("The provided unique job id")


@traceable
@tool("apollo-pinecone-gpt4-query", args_schema=Query)
def apollo_pinecone_gpt4(query: str):
    """
    A Pinecone vector database with external customer data
    """
    apollo = create_gpt4_pinecone_apollo_retriever()
    return apollo.run(query)


@traceable
@tool("discord-pinecone-gpt4-query", args_schema=Query)
def discord_pinecone_gpt4(query: str):
    """
    A Pinecone vector database with internal discord data
    """
    discord = create_gpt4_pinecone_discord_retriever()
    return discord.run(query)


@traceable
@tool("apollo-input-writer", args_schema=QueryWithJobId)
def apollo_input_writer(query: str, job_id: str) -> str:
    """
    Turn a natural language query into an Apollo input
    """
    query = create_apollo_input(query=query, job_id=job_id)
    return query["text"]


# Apollo Search Tool
class ApolloSearchInput(BaseConductorToolInput):
    person_titles: list[str] = Field(
        "An array of the person's title. Apollo will return results matching ANY of the titles passed in"
    )
    person_locations: list[str] = Field(
        'An array of strings denoting allowed locations of the person. Be sure to include city and country separated by a comma. Example: "San Francisco, US" or "London, GB"'
    )
    # q_keywords: Optional[str] = Field("A string of words over which we want to filter the results")
    # prospected_by_current_team: Optional[list[str]] = Field('An array of string booleans defining whether we want models prospected by current team or not. "no" means to look in net new database only, "yes" means to see saved contacts only')
    # person_seniorties: Optional[list[str]] = Field('An array of strings denoting the seniorities or levels')
    # contact_email_status: Optional[list[str]] = Field('An array of strings denoting the email status of the contact')
    # q_organization_domains: Optional[list[str]] = Field('An array of strings denoting the domains of the organization')
    # organization_locations: Optional[list[str]] = Field('An array of strings denoting allowed locations of organization headquarters of the person')


@traceable
@tool("apollo-person-search-tool", args_schema=ApolloSearchInput)
def apollo_person_search_context(
    job_id: str,
    person_titles: list[str],
    person_locations: list[str],
    # q_keywords: Optional[str] = None,
    # prospected_by_current_team: Optional[list[str]] = None,
    # person_seniorties: Optional[list[str]] = None,
    # contact_email_status: Optional[list[str]] = None,
    # q_organization_domains: Optional[list[str]] = None,
    # organization_locations: Optional[list[str]] = None
):
    """
    Apollo Person Search Tool that should be used with looking for people in a given industry or company
    Helpful when you need to identify people in a specific industry or company
    """
    return generate_apollo_person_search_context(
        job_id=job_id,
        person_titles=person_titles,
        person_locations=person_locations,
        save=True,
    )
