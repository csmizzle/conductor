"""
Langchain tools for Customer Intelligence API
"""
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
import requests
import os
import logging


logger = logging.getLogger(__name__)


# Apollo Search Tool
class ApolloSearchInput(BaseModel):
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


@tool("apollo-person-search-tool", args_schema=ApolloSearchInput, return_direct=True)
def apollo_person_search(
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
    Helful when you need to idenitfy people in a specific industry or company
    """
    response = requests.post(
        url="https://api.apollo.io/v1/mixed_people/search",
        json={
            "api_key": os.getenv("APOLLO_API_KEY"),
            # "q_organization_domains": '\n'.join(q_organization_domains),
            "page": 1,
            "per_page": 50,
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
        return response.json()
    else:
        logger.error(f"Failed to fetch data from Apollo: {response.status_code}")
        logger.error(f"Failed to fetch data from Apollo: {response.text}")
        return "Failed to fetch data from Apollo, I should use other data to answer the question."
