"""
Langchain tools for Customer Intelligence API
"""
from conductor.database.aws import upload_dict_to_s3
from conductor.llms import claude_v2_1
from conductor.prompts import CONDUCTOR_APOLLO_CUSTOMER_PROMPT
from conductor.models import BaseConductorToolInput
from conductor.parsers import (
    engagement_strategy_parser,
    EngagementStrategy,
    PersonEngagementStrategy,
)
from langchain.pydantic_v1 import Field
from langchain.tools import tool
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import requests
import os
import logging
from uuid import uuid4
import json


logger = logging.getLogger(__name__)

input_prompt = PromptTemplate(
    input_variables=["apollo_people_data"],
    template=CONDUCTOR_APOLLO_CUSTOMER_PROMPT,
    partial_variables={
        "format_instructions": engagement_strategy_parser.get_format_instructions()
    },
)


def create_engagement_strategy(apollo_people_data: str) -> EngagementStrategy:
    chain = LLMChain(
        llm=claude_v2_1,
        prompt=input_prompt,
    )
    response = chain.invoke({"apollo_people_data": apollo_people_data})
    return response


# Apollo Search Tool
class ApolloSearchInput(BaseConductorToolInput):
    person_titles: list[str] = Field(
        "An array of the person's title. Apollo will return results matching ANY of the titles passed in"
    )
    person_locations: list[str] = Field(
        'An array of strings denoting allowed locations of the person. Be sure to include city and country seperated by a comma. Example: "San Francisco, US" or "London, GB"'
    )
    # q_keywords: Optional[str] = Field("A string of words over which we want to filter the results")
    # prospected_by_current_team: Optional[list[str]] = Field('An array of string booleans defining whether we want models prospected by current team or not. "no" means to look in net new database only, "yes" means to see saved contacts only')
    # person_seniorties: Optional[list[str]] = Field('An array of strings denoting the seniorities or levels')
    # contact_email_status: Optional[list[str]] = Field('An array of strings denoting the email status of the contact')
    # q_organization_domains: Optional[list[str]] = Field('An array of strings denoting the domains of the organization')
    # organization_locations: Optional[list[str]] = Field('An array of strings denoting allowed locations of organization headquarters of the person')


@tool("apollo-person-search-tool", args_schema=ApolloSearchInput)
def apollo_person_search(
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
    response = requests.post(
        url="https://api.apollo.io/v1/mixed_people/search",
        json={
            "api_key": os.getenv("APOLLO_API_KEY"),
            # "q_organization_domains": '\n'.join(q_organization_domains),
            "page": 1,
            "per_page": 10,
            # "organization_locations": organization_locations,
            # "person_seniorities": person_seniorties,
            "person_titles": person_titles,
            "person_locations": person_locations,
            # "q_keywords" : q_keywords,
            # "prospected_by_current_team": prospected_by_current_team,
            # "contact_email_status": contact_email_status
        },
        headers={"Cache-Control": "no-cache", "Content-Type": "application/json"},
    )
    if response.ok:
        # store the data in s3
        file_id = uuid4()
        data = response.json()
        logger.info(
            f"Successfully fetched data from Apollo: {response.status_code} ..."
        )
        logger.info(
            f"Pushing raw Apollo response to s3: {os.getenv('APOLLO_S3_BUCKET')} ..."
        )
        json_response = json.dumps(data, indent=4)
        upload_dict_to_s3(
            data=json_response,
            bucket=os.getenv("APOLLO_S3_BUCKET"),
            key=f"{job_id}/{file_id}.json",
        )
        logger.info(
            "Creating an engagement strategy for each person in the results ..."
        )
        return f"People: {data["people"]}"


class PeopleInput(BaseConductorToolInput):
    people: list[dict] = Field(
        "A list of people JSON to create engagement strategies for"
    )


@tool("person-engagement-strategy-tool", args_schema=PeopleInput)
def person_engagement_strategy(job_id: str, people: list[dict]):
    """
    Use when you need to create an engagement strategy for a list of people
    """
    people: list[PersonEngagementStrategy] = []
    for person in people["people"]:
        print("Creating engagement strategy ...")
        engagement_strategy = create_engagement_strategy(person)
        engagement_strategy_object = engagement_strategy_parser.parse(
            engagement_strategy["text"]
        )
        people.append(
            PersonEngagementStrategy(
                person=person, engagement_strategy=engagement_strategy_object
            )
        )
    people_data = [object_.dict() for object_ in people]
    upload_dict_to_s3(
        data=json.dumps(people_data, indent=4),
        bucket=os.getenv("CONDUCTOR_S3_BUCKET"),
        key=f"{job_id}/engagement_strategy.json",
    )
    return people_data
