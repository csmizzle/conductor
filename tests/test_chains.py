"""
Test the chains module.
"""
from conductor.chains import (
    run_set_graph_chain,
    Graph,
    Timeline,
    QueryMatch,
    run_timeline_chain,
    run_query_match_chain,
)
from tests.constants import TEST_COMPLEX_NARRATIVE


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
