"""
Functions for Apollo data
"""
from conductor.chains import create_engagement_strategy
from conductor.parsers import engagement_strategy_parser, PersonEngagementStrategy
from conductor.context.apollo import (
    ApolloPersonSearchContext,
    ApolloPersonSearchRawContext,
)
from conductor.database.aws import upload_dict_to_s3
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
def create_apollo_engagement_strategies(data: dict) -> list[PersonEngagementStrategy]:
    """
    Use LLM chain to create tailored engagement strategies for each person
    """
    people = []
    context_creator = ApolloPersonSearchRawContext()
    contexts = context_creator.create_context(data=data)
    for idx, context in enumerate(contexts):
        engagement_strategy = create_engagement_strategy(context)
        engagement_strategy_object = engagement_strategy_parser.parse(
            engagement_strategy["text"]
        )
        people.append(
            PersonEngagementStrategy(
                person=data["people"][idx],
                engagement_strategy=engagement_strategy_object,
                context=context,
            )
        )
    return people


@traceable
def generate_apollo_person_search_job_context(
    job_id: str,
    person_titles: list[str],
    person_locations: list[str],
    raw_data_bucket: str,
    engagement_strategy_bucket: str,
    save: bool = True,
) -> str:
    people_data = apollo_api_person_search(
        person_titles=person_titles, person_locations=person_locations
    )
    if save:
        upload_dict_to_s3(
            data=people_data,
            bucket=raw_data_bucket,
            key=f"{job_id}/raw.json",
        )
    engagement_strategies = create_apollo_engagement_strategies(people_data)
    if len(engagement_strategies) > 0:
        dict_data = [
            engagement_strategy.dict() for engagement_strategy in engagement_strategies
        ]
        if save:
            upload_dict_to_s3(
                data=dict_data,
                bucket=engagement_strategy_bucket,
                key=f"{job_id}/apollo_person_search.json",
            )
        enriched_context_creator = ApolloPersonSearchContext()
        return f"Successfully ran Apollo Person Search Tool. Results {"\n".join(enriched_context_creator.create_context(data=dict_data))}"
    else:
        return "No results found for Apollo Person Search Tool."


@traceable
def generate_apollo_person_search_context(
    person_titles: list[str],
    person_locations: list[str],
) -> str:
    people_data = apollo_api_person_search(
        person_titles=person_titles, person_locations=person_locations
    )
    engagement_strategies = create_apollo_engagement_strategies(people_data)
    if len(engagement_strategies) > 0:
        dict_data = [
            engagement_strategy.dict() for engagement_strategy in engagement_strategies
        ]
        enriched_context_creator = ApolloPersonSearchContext()
        return f"Successfully ran Apollo Person Search Tool. Results {"\n".join(enriched_context_creator.create_context(data=dict_data))}"
    else:
        return "No results found for Apollo Person Search Tool."


@traceable
def generate_apollo_person_domain_search_context(
    company_domains: list[str],
) -> str:
    people_data = apollo_api_person_domain_search(company_domains=company_domains)
    if people_data:
        enriched_context_creator = ApolloPersonSearchRawContext()
        return f"Successfully ran Apollo Person Search Tool. Results {"\n".join(enriched_context_creator.create_context(data=people_data))}"
    else:
        return "No results found for Apollo Person Search Tool."


@traceable
def generate_apollo_person_domain_search_strategy_context(
    company_domains: list[str],
) -> str:
    people_data = apollo_api_person_domain_search(company_domains=company_domains)
    engagement_strategies = create_apollo_engagement_strategies(people_data)
    if len(engagement_strategies) > 0:
        dict_data = [
            engagement_strategy.dict() for engagement_strategy in engagement_strategies
        ]
        enriched_context_creator = ApolloPersonSearchContext()
        return f"Successfully ran Apollo Person Search Tool. Results {"\n".join(enriched_context_creator.create_context(data=dict_data))}"
    else:
        return "No results found for Apollo Person Search Tool."
