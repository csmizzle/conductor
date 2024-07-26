from pydantic.v1 import BaseModel, Field
from typing import Optional, List
from abc import ABC, abstractmethod
from enum import Enum
from textwrap import indent


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


class ReportTone(Enum):
    """
    Enum for report tone
    """

    PROFESSIONAL = "professional"
    INFORMAL = "informal"
    INFORMATIONAL = "informational"
    ANALYTICAL = "analytical"
    PERSUASIVE = "persuasive"
    CRITICAL = "critical"


class ReportPointOfView(Enum):
    """
    Enum for report point of view
    """

    FIRST_PERSON = "first person"
    THIRD_PERSON = "third person"


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


class SectionV2(BaseModel):
    title: str = Field(description="Title of report section")
    paragraphs: List[ParagraphV2] = Field(description="List of paragraphs")
    sources: Optional[List[str]] = Field(description="List of sources for the section")
    tone: Optional[ReportTone] = Field(description="Tone of the section")
    style: Optional[ReportStyle] = Field(description="Style of the section")
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
