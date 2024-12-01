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


def generate_relationship(source: str, target: str) -> str:
    """
    Generate a relationship type label that would make sense in a graph database.
    These are typically in the form of a verb or adjective and are used to describe the relationship between source and target nodes.
    """
    generate = dspy.Predict(GenerateRelationshipLabel)
    return generate(source=source, target=target).relationship
