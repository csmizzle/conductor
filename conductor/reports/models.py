from pydantic.v1 import BaseModel, Field
from typing import Optional, List
from abc import ABC, abstractmethod


class Paragraph(BaseModel):
    title: Optional[str] = Field(default="", description="Title of the paragraph")
    content: str = Field(description="Content of the paragraph")


class Section(BaseModel):
    title: str = Field(description="Title of report section")
    paragraphs: List[Paragraph] = Field(description="List of paragraphs")


class Report(BaseModel):
    title: str = Field(description="Title of the report")
    description: str = Field(description="Description of the report")
    sections: list[Section] = Field(description="Sections in the report")
    raw: Optional[str] = Field(description="Raw report")


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
