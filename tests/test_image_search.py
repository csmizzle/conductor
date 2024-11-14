"""
Test image search
"""
from conductor.images.signatures import RelationshipImageSearch
from conductor.images.search import (
    build_searches_from_graph,
    collect_images_from_graph,
    collect_images_from_queries,
)
from conductor.graph.models import Graph
import dspy
from tests.utils import load_model_from_test_data
from langtrace_python_sdk import with_langtrace_root_span
import os


mini = dspy.LM("gpt-4o-mini")


@with_langtrace_root_span()
def test_relationship_image_search() -> None:
    dspy.configure(lm=mini)
    graph = load_model_from_test_data(Graph, "graph.json")
    relationship_query = dspy.Predict(RelationshipImageSearch)
    query = relationship_query(
        relationship=graph.relationships[0],
    )
    assert isinstance(query, dspy.Prediction)
    assert isinstance(query.query, str)


@with_langtrace_root_span()
def test_build_searches_from_graph() -> None:
    dspy.configure(lm=mini)
    graph = load_model_from_test_data(Graph, "graph.json")
    search_queries = build_searches_from_graph(graph=graph)
    assert isinstance(search_queries, dict)
    assert len(search_queries) + 1 == len(graph.relationships)


def test_images_from_queries() -> None:
    dspy.configure(lm=mini)
    graph = load_model_from_test_data(Graph, "graph.json")
    search_queries = build_searches_from_graph(graph=graph)
    images = collect_images_from_queries(
        queries=search_queries, api_key=os.getenv("SERPAPI_API_KEY")
    )
    assert isinstance(search_queries, dict)
    assert len(search_queries) + 1 == len(graph.relationships)
    assert isinstance(images, list)


@with_langtrace_root_span()
def test_collect_images_from_graph() -> None:
    dspy.configure(lm=mini)
    graph = load_model_from_test_data(Graph, "graph.json")
    images = collect_images_from_graph(
        graph=graph, api_key=os.getenv("SERPAPI_API_KEY")
    )
    assert isinstance(images, list)
