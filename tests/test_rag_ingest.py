"""
Test the rag ingest components
"""
from conductor.rag.ingest import (
    ingest_webpage,
    describe_image_from_url,
    relationships_to_image_query,
    queries_to_image_results,
    describe_from_image_search_results,
)
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.llms import openai_gpt_4o
from tests.constants import GRAPH_JSON
from conductor.reports.models import Graph, RelationshipType
import pytest
import os


def test_ingest_webpage() -> None:
    """
    Test ingestion of a webpage
    """
    test_url = "https://flashpoint.io"
    webpage = ingest_webpage(url=test_url)
    assert isinstance(webpage, WebPage)


def test_ingest_webpage_invalid_url() -> None:
    """
    Test ingestion of a webpage with an invalid URL
    """
    test_url = "https://flashpoint.io/asdfasdf"
    with pytest.raises(Exception):
        ingest_webpage(url=test_url)


def test_ingest_image_from_url() -> None:
    """
    Test ingestion of an image from a URL
    """
    test_url = "https://assets.weforum.org/sf_account/image/-T6sEZZYrPjKBFqgJR9nhnbLpKoafHG__y0ZlbMJaU8.jpg"
    image = describe_image_from_url(
        image_url=test_url,
        model=openai_gpt_4o,
        metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
    )
    assert isinstance(image, SourcedImageDescription)


def test_ingest_image_from_url_invalid_url() -> None:
    """
    Test ingestion of an image from an invalid URL
    """
    test_url = "https://flashpoint.io/asdfasdf"
    with pytest.raises(Exception):
        describe_image_from_url(
            image_url=test_url,
            model=openai_gpt_4o,
            metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
        )


def test_graph_to_images() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    image_queries = relationships_to_image_query(
        graph=graph,
        relationship_types=[RelationshipType.FOUNDER.value],
        api_key=os.getenv("SERPAPI_API_KEY"),
    )
    assert isinstance(image_queries, list)
    image_search_results = queries_to_image_results(
        search_queries=image_queries, n_images=1
    )
    assert isinstance(image_search_results, list)
    assert len(image_search_results) == 2
    descriptions = describe_from_image_search_results(
        image_results=image_search_results, model=openai_gpt_4o
    )
    assert isinstance(descriptions, list)
    assert len(descriptions) == 1
