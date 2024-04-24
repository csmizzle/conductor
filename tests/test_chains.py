"""
Test different langchain chains
"""
from conductor.chains import create_apollo_input
from langsmith import unit


@unit
def test_create_apollo_input():
    query = "Find me a CEO in San Francisco"
    job_id = "12345"
    response = create_apollo_input(query, job_id)
    assert isinstance(response["text"], str)
