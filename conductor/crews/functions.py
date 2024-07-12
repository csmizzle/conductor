"""
Functions for Apollo data
"""
from conductor.crews.context import (
    ApolloPersonSearchRawContext,
)
import requests
from typing import Union
import os
from langsmith import traceable


def apollo_api_person_search(
    person_titles: list[str], person_locations: list[str], per_page: int = 3
) -> Union[str, None]:
    """
    Call Apollo API to search for people
    """
    response = requests.post(
        url="https://api.apollo.io/v1/mixed_people/search",
        json={
            "api_key": os.getenv("APOLLO_API_KEY"),
            # "q_organization_domains": '\n'.join(q_organization_domains),
            "page": 1,
            "per_page": per_page,
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


def apollo_api_person_domain_search(
    company_domains: list[str],
    per_page: int = 3,
) -> Union[str, None]:
    """
    Call Apollo API to search for companies
    """
    response = requests.post(
        url="https://api.apollo.io/v1/mixed_people/search",
        json={
            "api_key": os.getenv("APOLLO_API_KEY"),
            "q_organization_domains": "\n".join(company_domains),
            "page": 1,
            "per_page": per_page,
        },
        headers={"Cache-Control": "no-cache", "Content-Type": "application/json"},
    )
    if response.ok:
        return response.json()


@traceable
def generate_apollo_person_domain_search_context(
    company_domains: list[str], results: int = 3
) -> str:
    people_data = apollo_api_person_domain_search(
        company_domains=company_domains, per_page=results
    )
    if people_data:
        enriched_context_creator = ApolloPersonSearchRawContext()
        return f"Successfully ran Apollo Person Search Tool. Results {"\n".join(enriched_context_creator.create_context(data=people_data))}"
    else:
        return "No results found for Apollo Person Search Tool."
