"""
DSPY signatures for the conductor graph.
"""
import dspy
from conductor.graph import models


class RelationshipQuery(dspy.Signature):
    """
    You are an expert in looking at a triple comprised of subject, predicate, and object and translating into a natural language query.
    """

    specification: str = dspy.InputField(
        description="Specification for entity extraction", prefix="Specification: "
    )
    triple_type: models.TripleType = dspy.InputField(
        description="Triple containing source_type, relationship_type, and target_type",
        prefix="Triple: ",
    )
    query: str = dspy.OutputField(description="Generated query", prefix="Query: ")


class ExtractedRelationships(dspy.Signature):
    """
    Extract the entities and relationships from a document.
    Use the query to ground the extractions, they should only be relevant to the query.
    The triple serves are your grounding to extract the correct kinds of entities and relationships.
    Only extract the entities and relationships that are relevant to the triple and ignore the rest.
    The entity type, relationship type, and target type will only be one of the types in the triple_type.
    """

    query: str = dspy.InputField(
        description="Query to ground extraction", prefix="Query: "
    )
    triple_type: models.TripleType = dspy.InputField(
        description="Triple containing source_type, relationship_type, and target_type",
        prefix="Triple: ",
    )
    document: str = dspy.InputField(
        description="Document to extract relationships from", prefix="Documents: "
    )
    relationships: list[models.Relationship] = dspy.OutputField(
        description="Extracted relationships", prefix="Relationships: "
    )
