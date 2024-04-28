"""
Test different langchain chains
"""
from conductor.chains import create_apollo_input, create_html_summary
from tests.vars import TEST_HTML_DATA
from bs4 import BeautifulSoup
from langsmith import unit


@unit
def test_create_apollo_input() -> None:
    query = "Find me a CEO in San Francisco"
    job_id = "12345"
    response = create_apollo_input(query, job_id)
    assert isinstance(response["text"], str)


@unit
def test_create_html_summary() -> None:
    text = BeautifulSoup(TEST_HTML_DATA, "html.parser").get_text()
    response = create_html_summary(text)
    assert isinstance(response["text"], str)
