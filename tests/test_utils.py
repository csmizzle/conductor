from conductor.utils.graph import graph_to_networkx, draw_networkx
from conductor.utils.timeline import draw_timeline
from tests.constants import GRAPH_JSON, BASEDIR, TIMELINE_JSON
from conductor.chains.models import Graph, Timeline
from networkx import DiGraph
import os


def test_graph_to_pyvis() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    net = graph_to_networkx(graph)
    assert isinstance(net, DiGraph)


def test_draw_networkx() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    net = graph_to_networkx(graph)
    drawing = draw_networkx(net, os.path.join(BASEDIR, "data", "testgraph.png"))
    assert drawing


def test_draw_timeline() -> None:
    timeline = Timeline.model_validate(TIMELINE_JSON)
    drawing = draw_timeline(
        timeline, os.path.join(BASEDIR, "data", "test_timeline.png")
    )
    assert drawing
