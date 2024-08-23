from conductor.utils.graph import graph_to_networkx
from tests.constants import GRAPH_JSON
from conductor.chains.models import Graph
from networkx import DiGraph
import networkx as nx
import matplotlib.pyplot as plt


def test_graph_to_pyvis() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    net = graph_to_networkx(graph)
    assert isinstance(net, DiGraph)


def test_draw_networkx() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    net = graph_to_networkx(graph)
    d = dict(net.degree)
    nx.draw_spring(
        net,
        with_labels=True,
        nodelist=d,
        # node_size=[d[k]*300 for k in d]
    )
    plt.savefig("./testgraph.png", dpi=300, bbox_inches="tight")
    assert True
