from conductor.context.apollo import (
    ApolloPersonSearchContext,
    ApolloPersonSearchRawContext,
)
from tests.constants import TEST_APOLLO_DATA, TEST_APOLLO_RAW_DATA
from conductor.functions.apollo import apollo_api_person_domain_search


def test_apollo_person_search_context_engagement() -> None:
    person_context = ApolloPersonSearchContext().create_context(TEST_APOLLO_DATA)
    assert len(person_context) > 0
    assert person_context[0].startswith("Name: Grig B.")


def test_apollo_person_search_context_raw() -> None:
    person_context = ApolloPersonSearchRawContext().create_context(TEST_APOLLO_RAW_DATA)
    assert len(person_context) > 0
    assert person_context[0].startswith("Name: Grig B.")


def test_apollo_url_search_context_raw() -> None:
    test_domain = ["https://trssllc.com"]
    person_data = apollo_api_person_domain_search(company_domains=test_domain)
    enriched_context_creator = ApolloPersonSearchRawContext().create_context(
        data=person_data
    )
    assert len(enriched_context_creator) > 0
