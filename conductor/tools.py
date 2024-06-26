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
from conductor.parsers import GmailInput
from conductor.functions.apollo import (
    generate_apollo_person_search_context,
    generate_apollo_person_search_job_context,
)
from conductor.functions.google import (
    create_gmail_input_from_input,
    create_gmail_draft,
    send_gmail,
)
from conductor.chains import create_email_from_context
from conductor.chains import create_apollo_input_with_job_id, create_apollo_input
import logging
from langsmith import traceable
import os


logger = logging.getLogger(__name__)


class Query(BaseModel):
    query: str = Field("A natural language query")


class QueryWithJobId(Query):
    job_id: str = Field("The provided unique job id")


class ApolloEmailInput(BaseModel):
    to: list[str] = Field("A list of valid email addresses to send the email to")
    person_titles: list[str] = Field(
        "An array of the person's title. Apollo will return results matching ANY of the titles passed in"
    )
    person_locations: list[str] = Field(
        'An array of strings denoting allowed locations of the person. Be sure to include city and country separated by a comma. Example: "San Francisco, US" or "London, GB"'
    )


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


@tool("apollo-input-with-job-writer", args_schema=QueryWithJobId)
def apollo_input_with_job_writer(query: str, job_id: str) -> str:
    """
    Turn a natural language query into an Apollo input with a job id
    """
    query = create_apollo_input_with_job_id(query=query, job_id=job_id)
    return query["text"]


@tool("apollo-input-writer", args_schema=Query)
def apollo_input_writer(query: str) -> str:
    """
    Turn a natural language query into an Apollo input
    """
    query = create_apollo_input(query=query)
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


@tool("apollo-person-search-job-tool", args_schema=ApolloSearchInput)
def apollo_person_search_context_job(
    job_id: str,
    person_titles: list[str],
    person_locations: list[str],
    # q_keywords: Optional[str] = None,
    # prospected_by_current_team: Optional[list[str]] = None,
    # person_seniorties: Optional[list[str]] = None,
    # contact_email_status: Optional[list[str]] = None,
    # q_organization_domains: Optional[list[str]] = None,
    # organization_locations: Optional[list[str]] = None
) -> str:
    """
    Apollo Person Search Tool that should be used with looking for people in a given industry or company when also given a job id
    Helpful when you need to identify people in a specific industry or company
    """
    return generate_apollo_person_search_job_context(
        job_id=job_id,
        person_titles=person_titles,
        person_locations=person_locations,
        raw_data_bucket=os.getenv("APOLLO_S3_BUCKET"),
        engagement_strategy_bucket=os.getenv("CONDUCTOR_S3_BUCKET"),
        save=True,
    )


@tool("apollo-person-search-tool", args_schema=ApolloSearchInput)
def apollo_person_search_context(
    person_titles: list[str],
    person_locations: list[str],
    # q_keywords: Optional[str] = None,
    # prospected_by_current_team: Optional[list[str]] = None,
    # person_seniorties: Optional[list[str]] = None,
    # contact_email_status: Optional[list[str]] = None,
    # q_organization_domains: Optional[list[str]] = None,
    # organization_locations: Optional[list[str]] = None
) -> str:
    """
    Apollo Person Search Tool that should be used with looking for people in a given industry or company
    Helpful when you need to identify people in a specific industry or company
    """
    return generate_apollo_person_search_context(
        person_titles=person_titles,
        person_locations=person_locations,
    )


@tool("gmail-input-from-draft", args_schema=Query)
def gmail_input_from_input(input_: str) -> str:
    """
    Take natural language input and create a Gmail email input
    """
    parsed_input = create_gmail_input_from_input(input_)
    return f"Google Draft input: {parsed_input}"


@tool("gmail-draft", args_schema=GmailInput)
def gmail_draft(to: list[str], subject: str, message: str) -> str:
    """
    Create a Gmail draft
    """
    return create_gmail_draft(to=to, subject=subject, message=message)


@tool("gmail-send", args_schema=GmailInput)
def gmail_send(to: list[str], subject: str, message: str) -> str:
    """
    Create a Gmail draft
    """
    return send_gmail(to=to, subject=subject, message=message)


@tool("apollo-email-sender", args_schema=ApolloEmailInput)
def apollo_email_sender(
    to: list[str], person_titles: list[str], person_locations: list[str]
) -> str:
    """
    Use context created from Apollo Person Search to create and send an email.:
    """
    apollo_context = generate_apollo_person_search_context(
        person_titles=person_titles, person_locations=person_locations
    )
    email = create_email_from_context(
        context=apollo_context,
        sign_off="Best, John Envoy",
        tone="Summary for researcher",
    )
    sent_gmail = send_gmail(to=to, subject="Market Research", message=email["text"])
    return sent_gmail
