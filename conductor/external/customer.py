"""
Langchain tools for Customer Intelligence API
"""
from conductor.database.aws import upload_dict_to_s3
from conductor.llms import claude_v2_1
from conductor.prompts import CONDUCTOR_APOLLO_CUSTOMER_PROMPT
from conductor.models import BaseConductorToolInput
from conductor.parsers import customer_observation_parser, CustomerObservation
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
        "format_instructions": customer_observation_parser.get_format_instructions()
    },
)


def create_conductor_observation(apollo_people_data: str) -> CustomerObservation:
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


def clean_apollo_person_search(data: dict) -> str:
    """Clean Apollo data to be used in a LLM

    Args:
        data (dict): raw data from apollo

    Raises:
        NotImplementedError: _description_

    Returns:
        str: data in structured format for llm
    """
    observation = ""
    if "people" in data:
        for person in data["people"]:
            observation += f"Name: {person['name']}\n"
            if "is_likely_to_engage" in person:
                observation += f"Likely to Engage: {person['is_likely_to_engage']}\n"
            if "headline" in person:
                observation += f"Headline: {person['headline']}\n"
            if "title" in person:
                observation += f"Title: {person['title']}\n"
            if "seniority" in person:
                observation += f"Seniority: {person['seniority']}\n"
            if "website_url" in person["organization"]:
                observation += (
                    f"Company Website: {person["organization"]["website_url"]}\n"
                )
            if "linkedin_url" in person:
                observation += f"LinkedIn URL: {person['linkedin_url']}\n"
            if "twitter_url" in person:
                observation += f"Twitter URL: {person['twitter_url']}\n"
            if "github_url" in person:
                observation += f"Github URL: {person['github_url']}\n"
            if "facebook_url" in person:
                observation += f"Facebook URL: {person['facebook_url']}\n"
            if "city" in person:
                observation += f"City: {person['city']}\n"
            if "state" in person:
                observation += f"State: {person['state']}\n"
            if "country" in person:
                observation += f"Country: {person['country']}\n"
            if "departments" in person:
                observation += f"Departments: {' ,'.join(person['departments']) if person['departments'] else None}\n"
            if "subdepartments" in person:
                observation += f"Sub-Departments: {' ,'.join(person['subdepartments']) if person['subdepartments'] else None}\n"
            if "organization" in person:
                observation += "Organization:\n"
                if "name" in person["organization"]:
                    observation += f"Name: {person['organization']['name']}\n"
                if "primary_phone" in person["organization"]:
                    if "sanitized_number" in person["organization"]["primary_phone"]:
                        observation += f"Primary Phone: {person['organization']['primary_phone']['sanitized_number']}\n"
                if "website_url" in person["organization"]:
                    observation += f"Domain: {person['organization']['website_url']}\n"
                if "angellist_url" in person["organization"]:
                    observation += (
                        f"Angie List URL: {person['organization']['angellist_url']}\n"
                    )
                if "linkedin_url" in person["organization"]:
                    observation += (
                        f"LinkedIn URL: {person['organization']['linkedin_url']}\n"
                    )
                if "twitter_url" in person["organization"]:
                    observation += (
                        f"Twitter URL: {person['organization']['twitter_url']}\n"
                    )
                if "facebook_url" in person["organization"]:
                    observation += (
                        f"Facebook URL: {person['organization']['facebook_url']}\n"
                    )
            if "employment_history" in person:
                observation += "Employment History:\n"
                for idx in range(len(person["employment_history"])):
                    if "organization_name" in person["employment_history"][idx]:
                        observation += f"Company: {person['employment_history'][idx]['organization_name']}\n"
                    if "title" in person["employment_history"][idx]:
                        observation += (
                            f"Title: {person['employment_history'][idx]['title']}\n"
                        )
                    if "start_date" in person["employment_history"][idx]:
                        observation += f"Start Date: {person['employment_history'][idx]['start_date']}\n"
                    if "end_date" in person["employment_history"][idx]:
                        observation += f"End Date: {person['employment_history'][idx]['end_date']}\n"
                    if "description" in person["employment_history"][idx]:
                        observation += f"Description: {person['employment_history'][idx]['description']}\n"
            observation += "\n"
    return observation


@tool("apollo-person-search-tool", args_schema=ApolloSearchInput, return_direct=True)
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
        # cleaned_data = clean_apollo_person_search(data)
        # logger.info("Successfully cleaned apollo data")
        # logger.info(
        #     f"Pushing observation from cleaned Apollo response to s3: {os.getenv('APOLLO_S3_BUCKET')} ..."
        # )
        # upload_dict_to_s3(
        #     data=cleaned_data,
        #     bucket=os.getenv("APOLLO_S3_BUCKET"),
        #     key=f"{job_id}/{file_id}.txt",
        # )
        logger.info("Summarizing for conductor observation ...")
        observation = create_conductor_observation(json_response)
        customer_observation_object = customer_observation_parser.parse(
            observation["text"]
        )
        upload_dict_to_s3(
            data=customer_observation_object.json(indent=4),
            bucket=os.getenv("CONDUCTOR_S3_BUCKET"),
            key=f"{job_id}/apollo_person_search.json",
        )
        return observation["text"]
    else:
        logger.error(f"Failed to fetch data from Apollo: {response.status_code}")
        logger.error(f"Failed to fetch data from Apollo: {response.text}")
        return "Failed to fetch data from Apollo, I should use other data to answer the question."
