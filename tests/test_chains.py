"""
Test different langchain chains
"""
from conductor.chains import (
    create_apollo_input_with_job_id,
    create_html_summary,
    get_parsed_html_summary,
    map_reduce_summarize,
    create_apollo_input_structured,
)
from conductor.parsers import HtmlSummary, ApolloInput
from tests.vars import TEST_HTML_DATA
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
