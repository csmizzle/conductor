from crewai_tools.tools.base_tool import BaseTool
from serpapi import GoogleSearch
from pydantic.v1 import BaseModel, Field
from typing import Optional, Any, Type
from textwrap import dedent
import os


class FixedSerpSearchToolSchema(BaseModel):
    """Input for SerpSearchTool."""

    pass


class SerpSearchToolSchema(FixedSerpSearchToolSchema):
    """Input for SerpSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query that will retrieve search engine results page",
    )


class SerpSearchTool(BaseTool):
    name: str = "Search Engine Results Page (SERP) Tool"
    description: str = "A tool that can be used to scrape search engine results page (SERP) using a search query."
    args_schema: Type[BaseModel] = SerpSearchToolSchema

    def __init__(self, search_query: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if search_query is not None:
            self.args_schema = FixedSerpSearchToolSchema
            self.description = (
                "A tool that can be used to scrape search engine results page (SERP)."
            )
            self._generate_description()

    def _run(self, **kwargs: Any) -> Any:
        if os.getenv("SERPAPI_API_KEY") is None:
            raise ValueError("SERPAPI_API_KEY is not set in environment variables")
        search_context = []
        search_query = kwargs.get("search_query")
        search = GoogleSearch(
            {
                "q": search_query,
                "hl": "en",
                "gl": "us",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
        results_dict = search.get_dict()
        # add answer box to search context
        if "answer_box" in results_dict:
            search_context.append(
                dedent(
                    f"""
            Answer Box: {results_dict["answer_box"]["answer"] if "answer" in results_dict["answer_box"] else "No answer found"}
            Answer Box Source Link: {results_dict["answer_box"]["displayed_link"] if "displayed_link" in results_dict["answer_box"] else "No source link found"}
            """
                )
            )
        # add organic results to search context
        if "organic_results" in results_dict:
            for result in results_dict["organic_results"]:
                search_context.append(
                    dedent(
                        f"""
                Title: {result["title"]}
                Link: {result["link"]}
                Snippet: {result["snippet"]}
                """
                    )
                )
        return "\n".join(search_context)
