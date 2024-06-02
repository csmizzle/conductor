from pydantic import BaseModel
from abc import ABC, abstractmethod


class Paragraph(BaseModel):
    title: str
    content: str


class Report(BaseModel):
    title: str
    description: str
    paragraphs: list[Paragraph]


class Generator(ABC):
    @abstractmethod
    def generate(self) -> Report:
        pass
