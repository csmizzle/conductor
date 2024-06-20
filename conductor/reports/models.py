from pydantic.v1 import BaseModel, Field
from typing import Optional, List
from abc import ABC, abstractmethod
from enum import Enum


class ReportStyle(Enum):
    """
    Enum for report style
    """

    BULLETED = "as bulleted lists, avoiding long paragraphs."
    NARRATIVE = "as long form narratives, avoiding bullet points and short sentences."


class Paragraph(BaseModel):
    title: Optional[str] = Field(default="", description="Title of the paragraph")
    content: str = Field(description="Content of the paragraph")


class Section(BaseModel):
    title: str = Field(description="Title of report section")
    paragraphs: List[Paragraph] = Field(description="List of paragraphs")


class ParsedReport(BaseModel):
    title: str = Field(description="Title of the report")
    description: str = Field(description="Description of the report")
    sections: List[Section] = Field(description="Sections in the report")


class Report(BaseModel):
    report: Optional[ParsedReport] = Field(description="Parsed report")
    raw: Optional[str] = Field(description="Raw report")
    style: ReportStyle = Field(
        default=ReportStyle.BULLETED, description="Style of the report"
    )


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
