"""
Base models for conductor
"""
from typing import Any
from abc import ABC, abstractmethod
import jmespath


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
