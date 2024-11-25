"""
Evrim Graph Models
These models will allow for entity and relationship extraction from text to create a subgraph attached to the profile.
Key models:
    - Entity
    - Relationship
"""
from conductor.flow.rag import DocumentWithCredibility
from conductor.flow.credibility import SourceCredibility
from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """
    Entity types
    """

    PERSON = "PERSON"
    COMPANY = "COMPANY"


class RelationshipType(Enum):
    EMPLOYEE = "EMPLOYEE"
    DIRECTOR = "DIRECTOR"
    FOUNDER = "FOUNDER"
    EXECUTIVE = "EXECUTIVE"
    ASSOCIATED = "ASSOCIATED"
    PARENT_COMPANY = "PARENT_COMPANY"
    SUBSIDIARY = "SUBSIDIARY"
    ACQUIRED = "ACQUIRED"
    LOCATED_IN = "LOCATED_IN"


class Entity(BaseModel):
    """
    Entity model
    """

    entity_type: EntityType = Field(description="Entity type")
    name: str = Field(description="Entity value")

    class Config:
        use_enum_values = True


class Relationship(BaseModel):
    """
    Relationship model
    """

    source: Entity = Field(description="Source entity")
    target: Entity = Field(description="Target entity")
    relationship_type: RelationshipType = Field(description="Relationship type")
    faithfulness: int = Field(
        ge=1, le=5, description="The faithfulness of the relationship"
    )
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the relationship"
    )
    confidence: int = Field(
        ge=1, le=5, description="The confidence of the relationship"
    )

    class Config:
        use_enum_values = True


class Graph(BaseModel):
    """
    Graph model
    """

    entities: List[Entity] = Field(description="List of entities")
    relationships: List[Relationship] = Field(description="List of relationships")

    class Config:
        use_enum_values = True


class TripleType(BaseModel):
    """
    Triple model
    """

    source: EntityType = Field(description="Source entity type")
    relationship_type: RelationshipType = Field(description="Relationship type")
    target: EntityType = Field(description="Target entity type")

    class Config:
        use_enum_values = True


class CitedRelationshipWithCredibility(BaseModel):
    """
    Cited relationship model
    """

    source: Entity = Field(description="Source entity")
    target: Entity = Field(description="Target entity")
    relationship_type: str = Field(description="Relationship type")
    # individual relationship metadata
    relationship_reasoning: str = Field(
        description="The reasoning behind the relationship"
    )
    relationship_faithfulness: int = Field(
        ge=1, le=5, description="The faithfulness of the relationship"
    )
    relationship_factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the relationship"
    )
    relationship_confidence: int = Field(
        ge=1, le=5, description="The confidence of the relationship"
    )
    # relationship extraction metadata
    relationships_query: str = Field(
        description="The query used to generate the relationship"
    )
    # document collection metadata
    document_content: str = Field(
        description="The document used to generate the relationship"
    )
    document_source: str = Field(description="The source of the document")
    document_source_credibility: SourceCredibility = Field(
        description="The credibility of the sources"
    )
    document_source_credibility_reasoning: str = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True


class AggregatedCitedEntity(BaseModel):
    """
    Aggregated cited entity model
    """

    entity: Entity = Field(description="The entity")
    documents: List[DocumentWithCredibility] = Field(description="List of documents")

    class Config:
        use_enum_values = True


class AggregatedCitedRelationship(BaseModel):
    """
    Aggregated documents with a single relationship
    """

    source: Entity = Field(description="Source entity")
    target: Entity = Field(description="Target entity")
    relationship_type: str = Field(description="Relationship type")
    # individual relationship metadata
    relationship_reasoning: str = Field(
        description="The reasoning behind the relationship"
    )
    relationship_faithfulness: int = Field(
        ge=1, le=5, description="The faithfulness of the relationship"
    )
    relationship_factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the relationship"
    )
    relationship_confidence: int = Field(
        ge=1, le=5, description="The confidence of the relationship"
    )
    # relationship extraction metadata
    relationships_query: str = Field(
        description="The query used to generate the relationship"
    )
    documents: List[DocumentWithCredibility] = Field(description="List of documents")

    class Config:
        use_enum_values = True


class AggregatedCitedGraph(BaseModel):
    """
    Cited graph model
    """

    entities: List[AggregatedCitedEntity] = Field(description="List of cited entities")
    relationships: List[AggregatedCitedRelationship] = Field(
        description="List of cited relationships"
    )

    class Config:
        use_enum_values = True
