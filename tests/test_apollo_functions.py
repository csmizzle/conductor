"""
Test tool
"""
from conductor.functions.apollo import (
    apollo_api_person_search,
    apollo_api_person_domain_search,
    create_apollo_engagement_strategies,
    generate_apollo_person_search_job_context,
    generate_apollo_person_search_context,
    generate_apollo_person_domain_search_context,
)
from conductor.parsers import PersonEngagementStrategy
from tests.vars import (
    TEST_APOLLO_JOB_ID,
    TEST_RAW_DATA_BUCKET,
    TEST_ENGAGEMENT_STRATEGIES_BUCKET,
    TEST_APOLLO_RAW_DATA,
)
from langsmith import unit


def test_apollo_person_search():
    results = apollo_api_person_search(
        person_titles=["CEO", "CTO"],
        person_locations=["San Francisco, CA"],
    )
    assert isinstance(results, dict)


def test_apollo_person_domain_search():
    results = apollo_api_person_domain_search(company_domains=["trssllc.com"])
    assert isinstance(results, dict)


@unit
def test_create_apollo_engagement_strategy():
    engagement_strategies = create_apollo_engagement_strategies(TEST_APOLLO_RAW_DATA)
    assert isinstance(engagement_strategies, list) and len(engagement_strategies) == 3
    for engagement_strategy in engagement_strategies:
        assert isinstance(engagement_strategy, PersonEngagementStrategy)


@unit
def test_apollo_person_search_job_context():
    results = generate_apollo_person_search_job_context(
        job_id=TEST_APOLLO_JOB_ID,
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


@unit
def test_apollo_person_search_context():
    results = generate_apollo_person_search_context(
        person_titles=["CEO", "CTO"],
        person_locations=["San Francisco, CA"],
    )
    assert (
        len(results) > 0
        and isinstance(results, str)
        and results.startswith("Successfully")
    )


@unit
def test_apollo_person_domain_search_context():
    results = generate_apollo_person_domain_search_context(
        company_domains=["trssllc.com"]
    )
    assert (
        len(results) > 0
        and isinstance(results, str)
        and results.startswith("Successfully")
    )
