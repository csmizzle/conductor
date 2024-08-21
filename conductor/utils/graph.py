"""
Graph utilities
"""
from conductor.chains.models import Graph
from pyvis.network import Network
import networkx as nx


def graph_to_pyvis(graph: Graph) -> Network:
    """
    Convert graph to pyvis network
    """
    label_map = {}
    color_map = {
        "PERSON": "blue",
        "ORGANIZATION": "green",
        "LOCATION": "red",
    }
    network_x_object = nx.DiGraph()
    for idx in range(len(graph.entities)):
        if graph.entities[idx].name not in label_map:
            label_map[graph.entities[idx].name] = idx
            network_x_object.add_node(
                idx,
                label=graph.entities[idx].name,
                title=graph.entities[idx].name,
                color=color_map[graph.entities[idx].type],
            )
    for relationship in graph.relationships:
        network_x_object.add_edge(
            label_map[relationship.source.name],
            label_map[relationship.target.name],
            title=relationship.type,
            label=relationship.type,
        )
    net = Network(directed=True)
    net.from_nx(network_x_object)
    return net
