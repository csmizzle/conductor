"""
Functions for Apollo data
"""
from conductor.chains import create_engagement_strategy
from conductor.parsers import engagement_strategy_parser, PersonEngagementStrategy
from conductor.database.aws import upload_dict_to_s3
import requests
from typing import Union
import os
from langsmith import traceable


def apollo_api_person_search(
    person_titles: list[str], person_locations: list[str]
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
            "per_page": 3,
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


@traceable
def create_apollo_engagement_strategies(data: dict) -> list[PersonEngagementStrategy]:
    """
    Use LLM chain to create tailored engagement strategies for each person
    """
    people = []
    for person in data["people"]:
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
    return people


@traceable
def generate_apollo_person_search_context(
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
        return f"Successfully collected Apollo data for job: {job_id} \n People Data: {dict_data}"
    else:
        return f"Failed to collect Apollo data for job: {job_id}"
