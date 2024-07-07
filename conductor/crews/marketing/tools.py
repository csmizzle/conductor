from crewai_tools.tools.base_tool import BaseTool
from serpapi import GoogleSearch
from pydantic.v1 import BaseModel, Field
from typing import Optional, Any, Type
from textwrap import dedent
import os
from conductor.functions.apollo import generate_apollo_person_domain_search_context
from conductor.crews.marketing.utils import (
    send_request_proxy,
    send_request_proxy_with_cache,
    clean_html,
    send_request_with_cache,
)
from redis import Redis
import requests


CONTEXT_LIMIT = os.getenv("CONTEXT_LIMIT", 200000)


# Utility Functions
def check_context_limit(context: str) -> str:
    if len(context) > CONTEXT_LIMIT:
        return context[:CONTEXT_LIMIT]
    else:
        return context


# Tool Inputs
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


class FixedScrapeWebsiteToolSchema(BaseModel):
    """Input for ScrapeWebsiteTool."""

    pass


class ScrapeWebsiteToolSchema(FixedScrapeWebsiteToolSchema):
    """Input for ScrapeWebsiteTool."""

    website_url: str = Field(..., description="Mandatory website url to read the file")


class ApolloPersonDomainSearchToolSchema(FixedApolloPersonDomainSearchToolSchema):
    """Input for ApolloPersonDomainSearchTool."""

    company_domain: str = Field(
        ...,
        description="Company domain to search for people associated with the company",
    )


# Tools
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
        response = requests.request(
            url=url,
            method="GET",
            headers=self.headers,
            cookies=self.cookies if self.cookies else {},
            timeout=5,
        )
        if response.ok:
            return clean_html(response)
        else:
            return f"Error: Unable to fetch page content for {url}."

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


# Cache Tools
class ScrapePageCacheTool(BaseTool):
    """
    Scrape websites and cache the content
    """

    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = ScrapeWebsiteToolSchema
    website_url: Optional[str] = None
    cookies: Optional[dict] = None
    headers: Optional[dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(self, website_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedScrapeWebsiteToolSchema
            self._generate_description()
        self._cache = Redis.from_url(os.getenv("REDIS_TOOL_CACHE_URL"))

    def _run(self, **kwargs: Any) -> Any:
        url = kwargs.get("website_url")
        content = send_request_with_cache(
            url=url,
            method="GET",
            cache=self._cache,
            headers=self.headers,
            cookies=self.cookies,
        )
        return check_context_limit(content)


class SerpSearchCacheTool(SerpSearchTool):
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
        self._cache = Redis.from_url(os.getenv("REDIS_TOOL_CACHE_URL"))

    def _get_page_content(self, url: str) -> str:
        content = send_request_with_cache(
            url=url,
            method="GET",
            cache=self._cache,
            headers=self.headers,
            cookies=self.cookies,
        )
        return check_context_limit(content)


# OxyLabs Proxy Tools
class ScrapePageOxyLabsTool(BaseTool):
    """
    Scrape websites through OxyLabs Proxy
    """

    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = ScrapeWebsiteToolSchema
    website_url: Optional[str] = None
    cookies: Optional[dict] = None
    headers: Optional[dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(
        self,
        website_url: Optional[str] = None,
        cookies: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = FixedScrapeWebsiteToolSchema
            self._generate_description()
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        # try to request with oxylabs proxy first and if it fails, send normal request
        content = send_request_proxy(
            url=kwargs.get("website_url", self.website_url),
            method="GET",
            oxylabs_username=os.getenv("OXYLABS_USERNAME"),
            oxylabs_password=os.getenv("OXYLABS_PASSWORD"),
            headers=self.headers,
            cookies=self.cookies if self.cookies else {},
            timeout=5,
        )
        content = clean_html(content)
        return check_context_limit(content)


class SerpSearchOxyLabsTool(SerpSearchTool):
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
        # try to request with oxylabs proxy first and if it fails, send normal request
        response = send_request_proxy(
            url=url,
            method="GET",
            oxylabs_username=os.getenv("OXYLABS_USERNAME"),
            oxylabs_password=os.getenv("OXYLABS_PASSWORD"),
            headers=self.headers,
            cookies=self.cookies,
            timeout=5,
        )
        content = clean_html(response)
        return check_context_limit(content)


# Cached & OxyLabs version of the tools
class SerpSearchOxylabsCacheTool(SerpSearchTool):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = Redis.from_url(os.getenv("REDIS_TOOL_CACHE_URL"))

    def _get_page_content(self, url: str) -> str:
        # check cache for url
        content = send_request_proxy_with_cache(
            url=url,
            method="GET",
            oxylabs_username=os.getenv("OXYLABS_USERNAME"),
            oxylabs_password=os.getenv("OXYLABS_PASSWORD"),
            cache=self._cache,
            headers=self.headers,
            cookies=self.cookies,
            timeout=5,
        )
        return check_context_limit(content)


class ScrapePageOxylabsCacheTool(BaseTool):
    name: str = "Read website content"
    description: str = "A tool that can be used to read a website content."
    args_schema: Type[BaseModel] = ScrapeWebsiteToolSchema
    website_url: Optional[str] = None
    cookies: Optional[dict] = None
    headers: Optional[dict] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = Redis.from_url(os.getenv("REDIS_TOOL_CACHE_URL"))

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        url = kwargs.get("website_url")
        # check cache for url
        content = send_request_proxy_with_cache(
            url=url,
            method="GET",
            oxylabs_username=os.getenv("OXYLABS_USERNAME"),
            oxylabs_password=os.getenv("OXYLABS_PASSWORD"),
            cache=self._cache,
            headers=self.headers,
            cookies=self.cookies,
            timeout=5,
        )
        return check_context_limit(content)
