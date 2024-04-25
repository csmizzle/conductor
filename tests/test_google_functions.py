"""
Test Google functions
"""
from conductor.functions.google import (
    create_gmail_draft,
    create_gmail_input_from_input,
    create_gmail_draft_from_input,
)
from tests.vars import TEST_GMAIL_INPUT, TEST_GMAIL_CREW_PROMPT, TEST_APOLLO_DATA
from langsmith import expect, unit


@unit
def test_create_gmail_draft() -> None:
    draft = create_gmail_draft(**TEST_GMAIL_INPUT)
    assert isinstance(draft, str)


@unit
def test_create_gmail_input_from_input() -> None:
    parsed_inputs = create_gmail_input_from_input(TEST_GMAIL_CREW_PROMPT)
    assert isinstance(parsed_inputs, dict)
    assert "to" in parsed_inputs
    assert "subject" in parsed_inputs
    assert "message" in parsed_inputs
    expect(parsed_inputs["to"]).to_equal(TEST_GMAIL_INPUT["to"])
    expect(parsed_inputs["subject"]).to_equal(TEST_GMAIL_INPUT["subject"])
    expect(parsed_inputs["message"]).to_equal(TEST_GMAIL_INPUT["message"])


@unit
def test_create_gmail_draft_from_input() -> None:
    created_draft = create_gmail_draft_from_input(TEST_GMAIL_CREW_PROMPT)
    assert isinstance(created_draft, str)
    assert created_draft.startswith("Draft created.")


@unit
def test_create_gmail_input_from_apollo_input() -> None:
    apollo_input = (
        "Create an introduction email to this person: " + TEST_APOLLO_DATA[0]["context"]
    )
    created_apollo_gmail_input = create_gmail_input_from_input(apollo_input)
    assert isinstance(created_apollo_gmail_input, dict)
    assert "to" in created_apollo_gmail_input
    assert "subject" in created_apollo_gmail_input
    assert "message" in created_apollo_gmail_input
