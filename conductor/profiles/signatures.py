import dspy


class GenerateRelationshipLabel(dspy.Signature):
    """
    Generate a relationship type label that would make sense in a graph database.
    These are typically in the form of a verb or adjective and are used to describe the relationship between source and target nodes.
    """

    source: str = dspy.InputField(description="The source node type")
    target: str = dspy.InputField(description="The target node type")
    relationship: str = dspy.OutputField(
        description="A description of the relationship"
    )


class GenerateRelationshipQuery(dspy.Signature):
    """
    Use the source, target, and relationship type to generate a query that can be used search for additional information about the relationship using a search engine.
    The query should be a string that can be used as a search query in a search engine that would return relevant information about the relationship.
    """

    source: str = dspy.InputField(description="The source node type")
    target: str = dspy.InputField(description="The target node type")
    relationship_type: str = dspy.InputField(description="The relationship type")
    query: str = dspy.OutputField(description="The search query")


def generate_relationship(source: str, target: str) -> str:
    """
    Generate a relationship type label that would make sense in a graph database.
    These are typically in the form of a verb or adjective and are used to describe the relationship between source and target nodes.
    """
    generate = dspy.Predict(GenerateRelationshipLabel)
    return generate(source=source, target=target).relationship


def generate_relationship_specification(
    source: str, target: str, relationship_type: str
) -> str:
    """
    Use the source, target, and relationship type to generate a query that can be used search for additional information about the relationship using a search engine.
    """
    generate = dspy.Predict(GenerateRelationshipQuery)
    return generate(
        source=source, target=target, relationship_type=relationship_type
    ).query
