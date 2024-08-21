from conductor.utils.graph import graph_to_pyvis
from tests.constants import GRAPH_JSON
from conductor.chains.models import Graph
from pyvis.network import Network


def test_graph_to_pyvis() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    net = graph_to_pyvis(graph)
    assert isinstance(net, Network)
