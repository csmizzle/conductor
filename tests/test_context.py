from conductor.context.apollo import (
    ApolloPersonSearchContext,
    ApolloPersonSearchRawContext,
)
from tests.vars import TEST_APOLLO_DATA, TEST_APOLLO_RAW_DATA


def test_apollo_person_search_context_engagement() -> None:
    person_context = ApolloPersonSearchContext().create_context(TEST_APOLLO_DATA)
    assert len(person_context) > 0
    assert person_context[0].startswith("Name: Grig B.")


def test_apollo_person_search_context_raw() -> None:
    person_context = ApolloPersonSearchRawContext().create_context(TEST_APOLLO_RAW_DATA)
    assert len(person_context) > 0
    assert person_context[0].startswith("Name: Grig B.")
