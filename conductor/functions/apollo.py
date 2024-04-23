"""
Functions for Apollo data
"""
from conductor.chains import create_engagement_strategy
from conductor.parsers import engagement_strategy_parser, PersonEngagementStrategy
import requests
from typing import Union
import os


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


# def scrape_organization_page(data: dict):
#     """
#     Celery task to scrape organization page
#     """
#     if "organization" in data:
#         if "website_url" in data["organization"]:
#             scrape_page.delay(data["organization"]["website_url"])


# upload_dict_to_s3(
#     data=json.dumps(people_data, indent=4),
#     bucket=os.getenv("CONDUCTOR_S3_BUCKET"),
#     key=f"{job_id}/apollo_person_search.json",
# )
# # update apollo knowledge base
# print("Updating Apollo Knowledge Base ...")
# ApolloPineconeCreateDestroyPipeline().update(job_id)
# print("Successfully updated Apollo Knowledge Base ...")
# return f"Job ran successfully. Data stored in S3 with job_id: {job_id}"
