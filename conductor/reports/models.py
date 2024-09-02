from pydantic import BaseModel, Field
from typing import Optional, List
from abc import ABC, abstractmethod
from enum import Enum
from textwrap import indent


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

    def combine_description_metadata(self):
        """
        Combine the description and metadata
        """
        return f"{self.description} {self.metadata}"


# Image processing pipeline
class ImageResult(BaseModel):
    """
    Image search result model
    """

    original_url: str = Field(..., description="The original URL of the image")
    title: str = Field(..., description="The title of the image in the search results")
    caption: Optional[str] = Field(
        ..., description="The caption of the image in the search results"
    )


class ImageSearchResult(BaseModel):
    """
    Image search results model
    """

    query: str = Field(..., description="The search query")
    results: List[ImageResult] = Field(
        default=[], description="The image search results"
    )


# Query to paragraph matching
class QueryMatch(BaseModel):
    """
    Query match model
    """

    determination: str = Field(
        ..., description="The determination of the search query to the paragraph text"
    )


class ParagraphTemplate(BaseModel):
    """
    Template for a paragraph in a report
    """

    title: Optional[str] = Field(description="Title of the paragraph")
    content_template: str = Field(description="Template for section of the paragraph")


class SectionTemplate(BaseModel):
    """
    Template for a section in a report
    """

    title: str = Field(description="Title of the section")
    paragraphs: List[ParagraphTemplate] = Field(
        description="List of paragraph templates in the section"
    )


class ReportTemplate(BaseModel):
    """
    Report Template that can be defined for different reports
    """

    sections: List[SectionTemplate] = Field(
        description="Sections in the report template"
    )
    key_questions: Optional[List[str]] = Field(
        description="List of key questions for the report"
    )

    def to_prompt_string(self) -> str:
        """
        Convert report template to a prompt string
        """
        prompt_string = ""
        for idx in range(len(self.sections)):
            prompt_string += f"{idx + 1}. {self.sections[idx].title}\n"
            for paragraph in self.sections[idx].paragraphs:
                prompt_string += indent(f"{paragraph.title}\n", "  ")
                prompt_string += indent(f"{paragraph.content_template}\n", "    ")
        return prompt_string


class ReportTemplatePromptGenerator(ABC):
    sections: list[SectionTemplate]
    key_questions: Optional[List[str]]

    def generate(self) -> ReportTemplate:
        return ReportTemplate(
            sections=self.sections,
            key_questions=self.key_questions,
        ).to_prompt_string()


class ReportStyle(Enum):
    """
    Enum for report style
    """

    BULLETED = "as bulleted lists, avoiding long paragraphs."
    NARRATIVE = "as long form narratives, avoiding bullet points and short sentences."
    MIXED = (
        "as a mixture of long form narratives and bulleted lists when it makes sense"
    )


class ReportStyleV2(Enum):
    """
    Enum for report style
    """

    BULLETED = "BULLETED"
    NARRATIVE = "NARRATIVE"
    MIXED = "MIXED"


class ReportTone(Enum):
    """
    Enum for report tone
    """

    PROFESSIONAL = "PROFESSIONAL"
    INFORMAL = "INFORMAL"
    INFORMATIONAL = "INFORMATIONAL"
    ANALYTICAL = "ANALYTICAL"
    PERSUASIVE = "PERSUASIVE"
    CRITICAL = "CRITICAL"


class ReportPointOfView(Enum):
    """
    Enum for report point of view
    """

    FIRST_PERSON = "FIRST_PERSON"
    THIRD_PERSON = "THIRD_PERSON"


class Paragraph(BaseModel):
    title: Optional[str] = Field(default="", description="Title of the paragraph")
    content: str = Field(description="Content of the paragraph")


class Section(BaseModel):
    title: str = Field(description="Title of report section")
    paragraphs: List[Paragraph] = Field(description="List of paragraphs")


class ParagraphV2(BaseModel):
    title: Optional[str] = Field(
        default="", description="Title of the paragraph if needed"
    )
    sentences: list[str] = Field(
        description="List of sentences in the paragraph. Each element in the list must contain only one sentence."
    )
    images: Optional[ImageSearchResult] = Field(
        description="Image search results for the paragraph. This is to be left blank during the initial report generation therefore LLMs should not fill this field."
    )


class SectionV2(BaseModel):
    title: str = Field(description="Title of report section")
    paragraphs: List[ParagraphV2] = Field(description="List of paragraphs")
    sources: Optional[List[str]] = Field(description="List of sources for the section")
    tone: Optional[ReportTone] = Field(description="Tone of the section")
    style: Optional[ReportStyleV2] = Field(description="Style of the section")
    point_of_view: Optional[ReportPointOfView] = Field(
        description="Point of view of the section"
    )

    class Config:
        use_enum_values = True


class ParsedReport(BaseModel):
    title: str = Field(description="Title of the report")
    description: str = Field(description="Description of the report")
    sections: List[Section] = Field(description="Sections in the report")


class ParsedReportV2(BaseModel):
    title: str = Field(description="Title of the report")
    description: str = Field(description="Description of the report")
    sections: List[SectionV2] = Field(description="Sections in the report")


class ReportV2(BaseModel):
    report: Optional[ParsedReportV2] = Field(description="Parsed report")
    raw: list[str] = Field(description="Raw report sections")


class Report(BaseModel):
    report: Optional[ParsedReport] = Field(description="Parsed report")
    raw: Optional[str] = Field(description="Raw report")
    style: Optional[ReportStyle] = Field(
        default=ReportStyle.BULLETED, description="Style of the report"
    )


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
