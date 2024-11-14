"""
Image search signatures
"""
import dspy
from conductor.graph.models import Relationship


class RelationshipImageSearch(dspy.Signature):
    """
    Transform a source, relationship, and target into an keyword image search for a search engine like Google
    """

    relationship: Relationship = dspy.InputField(
        desc="The relationship between the source and target"
    )
    query: str = dspy.OutputField(desc="The search engine query")
