from conductor.chains import prompts
from conductor.llms import openai_gpt_4o
from conductor.chains.models import Graph, EntityType, RelationshipType, Timeline

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
