"""
Test different langchain chains
"""
from conductor.chains import (
    create_apollo_input_with_job_id,
    create_html_summary,
    get_parsed_html_summary,
    map_reduce_summarize,
    create_apollo_input_structured,
    create_email_from_context_structured,
)
from conductor.parsers import HtmlSummary, ApolloInput, EmailDraft
from tests.constants import TEST_HTML_DATA
from bs4 import BeautifulSoup
from langsmith import unit


@unit
def test_create_apollo_input() -> None:
    query = "Find me a CEO in San Francisco"
    job_id = "12345"
    response = create_apollo_input_with_job_id(query, job_id)
    assert isinstance(response["text"], str)


@unit
def test_create_html_summary() -> None:
    text = BeautifulSoup(TEST_HTML_DATA, "html.parser").get_text()
    response = create_html_summary(text)
    assert isinstance(response["text"], str)


@unit
def test_get_parsed_html_summary() -> None:
    text = BeautifulSoup(TEST_HTML_DATA, "html.parser").get_text()
    summary = get_parsed_html_summary(text)
    assert isinstance(summary, HtmlSummary)


@unit
def test_map_reduce_summarize() -> None:
    contents = [
        BeautifulSoup(TEST_HTML_DATA, "html.parser").get_text(),
        BeautifulSoup(TEST_HTML_DATA, "html.parser").get_text(),
    ]
    response = map_reduce_summarize(contents)
    assert isinstance(response, dict)
    assert isinstance(response["output_text"], str)


@unit
def test_create_apollo_input_structured() -> None:
    response = create_apollo_input_structured(
        "Find me a CEO in San Francisco",
    )
    assert isinstance(response, ApolloInput)


@unit
def test_create_email_from_context_structured() -> None:
    response = create_email_from_context_structured(
        "formal",
        """
Turtles All the Way Down: Frames & iFrames
Some older sites might still use frames to break up thier pages. Modern ones might be using iFrames to expose data. Learn about turtles as you scrape content inside frames.
Advanced Topics: Real World Challenges You'll Encounter
Scraping real websites, you're likely run into a number of common gotchas. Get practice with spoofing headers, handling logins & session cookies, finding CSRF tokens, and other common network errors.
        """,
        "Best regards,",
    )
    assert isinstance(response, EmailDraft)
