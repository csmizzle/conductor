"""
Test the chains module.
"""
from conductor.chains import run_set_graph_chain, Graph, Timeline, run_timeline_chain
from tests.constants import TEST_COMPLEX_NARRATIVE


def test_run_set_graph_chain() -> None:
    graph = run_set_graph_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(graph, Graph)


def test_run_timeline_chain() -> None:
    timeline = run_timeline_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(timeline, Timeline)
