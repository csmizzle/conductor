from pydantic.v1 import BaseModel, Field
from abc import ABC, abstractmethod


class Paragraph(BaseModel):
    title: str = Field("Title of the paragraph")
    content: str = Field("Content of the paragraph")


class Section(BaseModel):
    title: str = Field("Title of the section")
    paragraphs: list[Paragraph] = Field("Paragraphs in the section")


class Report(BaseModel):
    title: str = Field("Title of the report")
    description: str = Field("Description of the report")
    sections: list[Section] = Field("Sections in the report")


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
