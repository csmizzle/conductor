"""
Test tool
"""
from conductor.functions.apollo import apollo_api_person_search


def test_apollo_person_search():
    results = apollo_api_person_search(
        person_titles=["CEO", "CTO"],
        person_locations=["San Francisco, CA"],
    )
    assert isinstance(results, dict)
