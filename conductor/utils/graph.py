"""
Graph utilities
"""
from conductor.chains.models import Graph
import networkx as nx


def graph_to_networkx(graph: Graph) -> nx.DiGraph:
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
                graph.entities[idx].name,
                label=graph.entities[idx].name,
                title=graph.entities[idx].name,
                color=color_map[graph.entities[idx].type],
            )
    for relationship in graph.relationships:
        network_x_object.add_edge(
            relationship.source.name,
            relationship.target.name,
            title=relationship.type,
            label=relationship.type,
        )
    return network_x_object
