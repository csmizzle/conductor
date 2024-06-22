from crewai_tools.tools.base_tool import BaseTool
from serpapi import GoogleSearch
from pydantic.v1 import BaseModel, Field
from typing import Optional, Any, Type
from textwrap import dedent
import os
import requests
from bs4 import BeautifulSoup
from conductor.functions.apollo import generate_apollo_person_domain_search_context
import re


CONTEXT_LIMIT = os.getenv("CONTEXT_LIMIT", 200000)


def check_context_limit(context: str) -> str:
    if len(context) > CONTEXT_LIMIT:
        return context[:CONTEXT_LIMIT]
    else:
        return context


class FixedSerpSearchToolSchema(BaseModel):
    """Input for SerpSearchTool."""

    pass


class SerpSearchToolSchema(FixedSerpSearchToolSchema):
    """Input for SerpSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query that will retrieve search engine results page",
    )


class FixedApolloPersonDomainSearchToolSchema(BaseModel):
    """Input for ApolloPersonDomainSearchTool."""

    pass


class ApolloPersonDomainSearchToolSchema(FixedApolloPersonDomainSearchToolSchema):
    """Input for ApolloPersonDomainSearchTool."""

    company_domain: str = Field(
        ...,
        description="Company domain to search for people associated with the company",
    )


class SerpSearchTool(BaseTool):
    name: str = "Search Engine Results Page (SERP) Tool"
    description: str = "A tool that can be used to scrape search engine results page (SERP) using a search query."
    args_schema: Type[BaseModel] = SerpSearchToolSchema
    cookies: Optional[dict] = None
    headers: Optional[dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(self, search_query: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if search_query is not None:
            self.args_schema = FixedSerpSearchToolSchema
            self.description = (
                "A tool that can be used to scrape search engine results page (SERP)."
            )
            self._generate_description()

    def _get_page_content(self, url: str) -> str:
        page = requests.get(
            url,
            timeout=15,
            headers=self.headers,
            cookies=self.cookies if self.cookies else {},
        )
        parsed = BeautifulSoup(page.content, "html.parser", from_encoding="iso-8859-1")
        text = parsed.get_text()
        text = "\n".join([i for i in text.split("\n") if i.strip() != ""])
        text = " ".join([i for i in text.split(" ") if i.strip() != ""])
        # remove all the special characters using regex
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

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
                page_content = self._get_page_content(result["link"])
                search_context.append(
                    dedent(
                        f"""
                Title: {result["title"]}
                Link: {result["link"]}
                Snippet: {result["snippet"]}
                Content: {page_content}
                """
                    )
                )
        return check_context_limit("\n".join(search_context))


class ApolloPersonDomainSearchTool(BaseTool):
    """
    Find people and their contact information associated with a company domain
    """

    name: str = "Apollo Person Domain Search"
    description: str = "A tool that can be used to find people and their contact information associated with a company domain."
    args_schema: Type[BaseModel] = ApolloPersonDomainSearchToolSchema

    def __init__(self, company_domain: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if company_domain is not None:
            self.args_schema = FixedApolloPersonDomainSearchToolSchema
            self.description = "A tool that can be used to find people and their contact information associated with a company domain."
            self._generate_description()

    def _run(self, **kwargs: Any) -> Any:
        company_domain = kwargs.get("company_domain")
        return generate_apollo_person_domain_search_context(
            company_domains=[company_domain], results=10
        )
