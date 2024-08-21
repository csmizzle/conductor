from conductor.chains import prompts
from conductor.chains.tools import image_search
from conductor.llms import openai_gpt_4o
from conductor.chains.models import Graph, EntityType, RelationshipType, Timeline
from tqdm import tqdm

graph_chain = prompts.graph_extraction_prompt | openai_gpt_4o | prompts.graph_parser
timeline_chain = (
    prompts.timeline_extraction_prompt | openai_gpt_4o | prompts.timeline_parser
)


def run_graph_chain(
    entity_types: list[str], relationship_types: list[str], text: str
) -> Graph:
    """Extract entities and relationships from a given text and create a relational graph.

    Args:
        entity_types (list[str]): entity types to extract
        relationship_types (list[str]): relationship types to extract
        text (str): text to extract entities and relationships from

    Returns:
        Graph: graph object containing the extracted entities and relationships
    """
    return graph_chain.invoke(
        dict(
            entity_types=entity_types,
            relationship_types=relationship_types,
            text=text,
        )
    )


def run_set_graph_chain(text: str) -> Graph:
    """Extract entities and relationships from a given text and create a relational graph.

    Args:
        text (str): text to extract entities and relationships from

    Returns:
        Graph: graph object containing the extracted entities and relationships
    """
    return run_graph_chain(
        text=text,
        entity_types=[enum.value for enum in EntityType],
        relationship_types=[enum.value for enum in RelationshipType],
    )


def run_timeline_chain(text: str) -> Timeline:
    """Extract events from a given text and create a timeline.

    Args:
        text (str): text to extract events from

    Returns:
        Timeline: timeline object containing the extracted events
    """
    return timeline_chain.invoke(
        dict(
            text=text,
        )
    )


# relationship to image search
def relationships_to_image_query(
    graph: Graph,
    api_key: str,
    relationship_types: list[RelationshipType] = None,
) -> list[dict]:
    """
    Converts a relationship to an image search.

    Args:
        graph (Graph): The graph to convert to an image search.
        api_key (str): The API key for the image search.
        relationship_types (list[RelationshipType]): The relationship types to convert to an image search. If None, all relationships are converted.

    Returns:
        str: The image search.
    """
    # iterate through graph and collect relations
    searches = set()
    for relationship in graph.relationships:
        # filter relationships based on relationship types
        if relationship_types:
            if relationship.type in relationship_types:
                # concat source and target and add to searches
                searches.add(relationship.source.name + " " + relationship.target.name)
        # add all relationships
        else:
            searches.add(relationship.source.name + " " + relationship.target.name)
    # iterate through searches and create image search
    results = []
    for search in tqdm(searches):
        results.append(
            image_search(
                query=search,
                api_key=api_key,
            )
        )
    return results
