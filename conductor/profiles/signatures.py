from conductor.profiles.models import RelationshipEvaluation
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


class RelationshipEvaluationSignature(dspy.Signature):
    """
    Evaluate the quality of a relationship between two entities.
    The quality should be a float between 1 and 5, where 1 is low quality and 5 is high quality.
    Low fidelity source and target names should lower the quality score.
    High fidelity source and target names should raise the quality score.
    Names that don't match the expected entity type should lower the quality score.
    Names that match the expected entity type should raise the quality score.
    Source and target names that do not match the relationship type should lower the quality score.
    Source and target names that match the relationship type should raise the quality score.
    """

    source_name: str = dspy.InputField(description="The name of the source entity")
    source_type: str = dspy.InputField(description="The type of the source entity")
    target_name: str = dspy.InputField(description="The name of the target entity")
    target_type: str = dspy.InputField(description="The type of the target entity")
    relationship_type: str = dspy.InputField(description="The type of relationship")
    quality: RelationshipEvaluation = dspy.OutputField(
        description="The quality of the relationship"
    )


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


def evaluate_extraction(
    source_name: str,
    source_type: str,
    target_name: str,
    target_type: str,
    relationship_type: str,
) -> RelationshipEvaluation:
    """
    Evaluate the quality of a relationship between two entities.
    """
    evaluate = dspy.ChainOfThought(RelationshipEvaluationSignature)
    return evaluate(
        source_name=source_name,
        source_type=source_type,
        target_name=target_name,
        target_type=target_type,
        relationship_type=relationship_type,
    ).quality
