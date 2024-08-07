"""
Base models for conductor
"""
from typing import Any, Optional
from abc import ABC, abstractmethod
import jmespath
from crewai.task import Task


class Context(ABC):
    """
    Context for a given job
    """

    def path_search(self, search: str, data: Any) -> str:
        expression = jmespath.compile(search)
        return expression.search(data)

    @abstractmethod
    def create_context(self, data: Any) -> list[str]:
        pass


class NamedTask(Task):
    name: str
    section_name: Optional[str] = None
