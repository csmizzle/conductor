"""
Graph utilities
"""
from conductor.reports.models import Graph
import networkx as nx
import matplotlib.pyplot as plt


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


def draw_networkx(network: nx.Graph, file_path: str) -> bool:
    """
    Draw networkx graph
    """
    d = dict(network.degree)
    plt.figure(figsize=(12, 8))
    pos = nx.circular_layout(network)
    nx.draw(
        network,
        pos,
        with_labels=True,
        node_color="skyblue",
        nodelist=d,
    )
    edge_labels = nx.get_edge_attributes(network, "label")
    nx.draw_networkx_edge_labels(
        network, pos, edge_labels=edge_labels, font_color="green"
    )
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    return True
