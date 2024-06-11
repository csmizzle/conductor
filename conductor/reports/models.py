from pydantic.v1 import BaseModel, Field
from abc import ABC, abstractmethod


class Paragraph(BaseModel):
    title: str = Field("Title of the paragraph")
    content: str = Field("Content of the paragraph")


class Report(BaseModel):
    title: str = Field("Title of the report")
    description: str = Field("Description of the report")
    paragraphs: list[Paragraph] = Field("Paragraphs in the report")


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
