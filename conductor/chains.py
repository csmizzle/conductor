from conductor.prompts import (
    input_prompt,
    apollo_input_prompt,
    gmail_input_prompt,
    html_summary_prompt,
)
from conductor.llms import claude_v2_1
from conductor.parsers import EngagementStrategy
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langsmith import traceable


@traceable
def create_conductor_search(
    job_id: str, geography: str, titles: list[str], industries: list[str]
) -> str:
    """Generate a search input for a conductor job

    Args:
        job_id (str): Conductor job id
        geography (str): Geography to search
        titles (list[str]): Titles to search
        industries (list[str]): Industries to search

    Returns:
        str: agent query
    """
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
        prompt=input_prompt,
    )
    response = chain.run(
        job_id=job_id, geography=geography, titles=titles, industries=industries
    )
    return response


@traceable
def create_engagement_strategy(apollo_people_data: str) -> EngagementStrategy:
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=input_prompt,
    )
    response = chain.invoke({"apollo_people_data": apollo_people_data})
    return response


@traceable
def create_apollo_input(query: str, job_id: str) -> str:
    """
    Extract Apollo input parameters from a general input string
    """
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
        prompt=apollo_input_prompt,
    )
    response = chain.invoke({"general_input": query, "job_id": job_id})
    return response


@traceable
def create_gmail_input(input_: str) -> str:
    """
    Extract Gmail input parameters from a general input string
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=gmail_input_prompt,
    )
    response = chain.invoke({"general_input": input_})
    return response


@traceable
def create_html_summary(raw: str) -> str:
    """
    Summarize the HTML content of a web page
    """
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=html_summary_prompt,
    )
    response = chain.invoke({"raw": raw})
    return response
