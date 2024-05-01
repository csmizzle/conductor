"""
Test email functions
"""
from langsmith import unit
from conductor.chains import create_email_from_context
from conductor.context.apollo import ApolloPersonSearchContext
from tests.vars import TEST_EMAIL_CONTEXT, TEST_APOLLO_DATA


@unit
def test_write_email_from_apollo_context() -> None:
    person_context = ApolloPersonSearchContext().create_context(TEST_APOLLO_DATA)
    assert len(person_context) > 0
    assert person_context[0].startswith("Name: Grig B.")
    email = create_email_from_context(
        context=person_context[0],
        sign_off="Best, John Envoy",
        tone="Summary for researcher",
    )
    assert isinstance(email["text"], str)
    assert email["text"].endswith("John Envoy")


@unit
def test_create_email_from_context() -> None:
    email = create_email_from_context(
        context=TEST_EMAIL_CONTEXT,
        sign_off="Best, John Envoy",
        tone="Summary for researcher",
    )
    assert isinstance(email["text"], str)
    assert email["text"].endswith("John Envoy")
