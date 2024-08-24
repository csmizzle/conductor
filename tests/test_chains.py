"""
Test the chains module.
"""
from conductor.chains import (
    run_set_graph_chain,
    Graph,
    Timeline,
    QueryMatch,
    RelationshipType,
    run_timeline_chain,
    run_query_match_chain,
    match_queries_to_paragraphs,
)
from conductor.rag.ingest import relationships_to_image_query, queries_to_image_results
from conductor.reports.models import ReportV2
from tests.constants import TEST_COMPLEX_NARRATIVE, GRAPH_JSON, REPORT_V2_JSON
import os
import json


def test_run_set_graph_chain() -> None:
    graph = run_set_graph_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(graph, Graph)


def test_run_timeline_chain() -> None:
    timeline = run_timeline_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(timeline, Timeline)


def test_run_query_match_chain() -> None:
    query_match = run_query_match_chain(
        search_query="John Fallone Fallone SV",
        text="""
        At the helm of Fallone SV is John Fallone, who serves as the Attorney and Managing Member. With
        over a decade of experience as a startup founder and legal counsel, John brings a wealth of knowledge
        to the firm. He co-founded SendHub, a startup that successfully raised approximately $10 million before
        its acquisition. John's academic credentials include a J.D. from George Mason Law School and an
        LL.M from Duke Law. Licensed in both California and North Carolina, he assists startups with venture
        financing, corporate structuring, contracts, mergers, and acquisitions. John is also a member of the
        """,
    )
    assert isinstance(query_match, QueryMatch)
    assert query_match.determination == "RELEVANT"


def test_match_queries_to_paragraphs() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    image_queries = relationships_to_image_query(
        graph=graph,
        relationship_types=[
            RelationshipType.FOUNDER.value,
            RelationshipType.EMPLOYEE.value,
        ],
        api_key=os.getenv("SERPAPI_API_KEY"),
    )
    assert isinstance(image_queries, list)
    image_search_results = queries_to_image_results(
        search_queries=image_queries, n_images=1
    )
    assert isinstance(image_search_results, list)
    assert len(image_search_results) == 7
    matched_paragraphs = match_queries_to_paragraphs(
        image_search_results=image_search_results,
        sections_filter=["Company Structure", "Personnel"],
        report=ReportV2.model_validate(REPORT_V2_JSON),
    )
    with open("tests/data/matched_paragraphs.json", "w") as f:
        json.dump(matched_paragraphs.model_dump(), f, indent=4)
    assert isinstance(matched_paragraphs, ReportV2)
