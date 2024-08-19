from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


# entity extraction
class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"


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

    name: str = Field(..., description="The name of the entity")
    type: EntityType = Field(..., description="The type of the entity")
    reason: str = Field(..., description="The reason for the entity label")

    class Config:
        use_enum_values = True


class Relationship(BaseModel):
    """
    Relationship model
    """

    source: Entity = Field(..., description="The source entity")
    target: Entity = Field(..., description="The target entity")
    type: RelationshipType = Field(..., description="The type of the relationship")
    reason: str = Field(
        ...,
        description="The reasoning for the relationship between the source and target entities",
    )

    class Config:
        use_enum_values = True


class Graph(BaseModel):
    """
    Graph model
    """

    entities: List[Entity] = Field(..., description="The entities in the graph")
    relationships: List[Relationship] = Field(
        ..., description="The relationships in the graph"
    )


class SourcedGraph(BaseModel):
    """
    Extracted graph model with source
    """

    graph: Graph = Field(..., description="The extracted graph")
    source: str = Field(..., description="The source url of the text")


# Timeline extraction
class TimelineEvent(BaseModel):
    """
    Timeline event model
    """

    year: str = Field(..., description="The year of the event in YYYY format")
    month: Optional[str] = Field(..., description="The date of the event in MM format")
    day: Optional[str] = Field(..., description="The day of the event in DD format")
    event: str = Field(..., description="Short event description")


class Timeline(BaseModel):
    """
    Timeline model
    """

    events: List[TimelineEvent] = Field(..., description="The events in the timeline")


# Image processing
class ImageDescription(BaseModel):
    """
    Image description model
    """

    description: str = Field(..., description="The description of the image")
    metadata: Optional[str] = Field(
        ..., description="The metadata provided with the image"
    )
    answer: str = Field(
        ..., description="The answer combining the description and metadata"
    )
