"""
Test tool
"""
from conductor.functions.apollo import (
    apollo_api_person_search,
    create_apollo_engagement_strategies,
    generate_apollo_person_search_context,
)
from conductor.parsers import PersonEngagementStrategy
from tests.vars import (
    TEST_JOB_ID,
    TEST_RAW_DATA_BUCKET,
    TEST_ENGAGEMENT_STRATEGIES_BUCKET,
)
from langsmith import unit


test_people_data = {
    "people": [
        {
            "name": "John Doe",
            "title": "CEO",
            "location": "San Francisco, CA",
        },
        {
            "name": "Jane Doe",
            "title": "CTO",
            "location": "San Francisco, CA",
        },
    ]
}


def test_apollo_person_search():
    results = apollo_api_person_search(
        person_titles=["CEO", "CTO"],
        person_locations=["San Francisco, CA"],
    )
    assert isinstance(results, dict)


@unit
def test_create_apollo_engagement_strategy():
    engagement_strategies = create_apollo_engagement_strategies(test_people_data)
    assert isinstance(engagement_strategies, list) and len(engagement_strategies) == 2
    for engagement_strategy in engagement_strategies:
        assert isinstance(engagement_strategy, PersonEngagementStrategy)


@unit
def test_apollo_person_search_context():
    results = generate_apollo_person_search_context(
        job_id=TEST_JOB_ID,
        raw_data_bucket=TEST_RAW_DATA_BUCKET,
        engagement_strategy_bucket=TEST_ENGAGEMENT_STRATEGIES_BUCKET,
        person_titles=["CEO", "CTO"],
        person_locations=["San Francisco, CA"],
    )
    assert (
        len(results) > 0
        and isinstance(results, str)
        and results.startswith("Successfully")
    )
