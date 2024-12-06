"""
Add two different kinds of tools
- Scrape into a vector database
- Search a vector database
"""
from crewai_tools.tools import ScrapeWebsiteTool
from crewai_tools.tools.base_tool import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Any, Type
from conductor.crews.marketing.tools import (
    ScrapeWebsiteToolSchema,
    SerpSearchToolSchema,
)
from conductor.rag.client import ElasticsearchRetrieverClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.rag.ingest import ingest, parallel_ingest
from conductor.rag.utils import (
    get_page_content_with_source_url,
    get_content_and_source_from_response,
)
from serpapi import GoogleSearch
import os
from loguru import logger


class FixedVectorSearchToolSchema(BaseModel):
    """Input for SerpSearchTool."""

    pass


class FixedVectorMetaSearchToolSchema(BaseModel):
    """Input for SerpSearchTool."""

    pass


class VectorSearchToolSchema(FixedVectorSearchToolSchema):
    """Input for SerpSearchTool."""

    search_query: str = Field(
        ...,
        description="A question that will be used to search the vector database.",
    )


class VectorMetaSearchToolSchema(FixedVectorMetaSearchToolSchema):
    """Input for search Elasticsearch by a single URL metadata field."""

    url: str = Field(
        ...,
        description="The URL of the company to find the website in vector database in the document metadata",
    )


class ScrapeWebsiteIngestTool(ScrapeWebsiteTool):
    name: str = "Get website content to add to the vector database"
    description: str = "A tool that can be used to read a website content. Returns confirmation that the data has been collected and ingested into the vector database or already exists."
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
        elasticsearch: Elasticsearch,
        index_name: str,
        website_url: Optional[str] = None,
        cookies: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_database = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=BedrockEmbeddings(),
            index_name=index_name,
        )
        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = ScrapeWebsiteToolSchema
            self._generate_description()
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        # check if documents already in vector database by looking at the metadata URL
        return ingest(
            client=self._vector_database,
            url=kwargs.get("website_url", self.website_url),
            headers=self.headers,
            cookies=self.cookies,
        )


class ScrapeWebsiteWithContentIngestTool(ScrapeWebsiteTool):
    name: str = "Get website content to add to the vector database"
    description: str = "A tool that can be used to read a website content and ingest into vector database. Returns the content of the website."
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
        elasticsearch: Elasticsearch,
        index_name: str,
        website_url: Optional[str] = None,
        cookies: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_database = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=BedrockEmbeddings(),
            index_name=index_name,
        )
        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"A tool that can be used to read {website_url}'s content."
            )
            self.args_schema = ScrapeWebsiteToolSchema
            self._generate_description()
            if cookies is not None:
                self.cookies = {cookies["name"]: os.getenv(cookies["value"])}

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        # check if documents already in vector database by looking at the metadata URL
        url = kwargs.get("website_url", self.website_url)
        ingested_content = ingest(
            client=self._vector_database,
            url=url,
            headers=self.headers,
            cookies=self.cookies,
        )
        if ingested_content:
            try:
                logger.info("Getting content from the vector database ...")
                data = self._vector_database.find_document_by_url(url=url)
                # return the text of the first document
                return get_content_and_source_from_response(data)
            except Exception as e:
                logger.exception(e)


class VectorSearchTool(BaseTool):
    """
    Search a vector database for relevant information.
    """

    name: str = "Search the vector database for relevant information."
    description: str = (
        "A tool that can be used to search a vector database for relevant information."
    )
    args_schema: Type[BaseModel] = VectorSearchToolSchema
    search_query: Optional[str] = None

    def __init__(
        self,
        elasticsearch: Elasticsearch,
        index_name: str,
        search_query: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._vector_database = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=BedrockEmbeddings(),
            index_name=index_name,
        )
        if search_query is not None:
            self.search_query = search_query
            self.description = f"A tool that can be used answer '{search_query}' using vectors in a vector database."
            self.args_schema = VectorSearchToolSchema
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        search_query = kwargs.get("search_query", self.search_query)
        documents = self._vector_database.store.similarity_search(
            query=search_query, k=5
        )
        return "\n".join(
            [get_page_content_with_source_url(document) for document in documents]
        )


class VectorSearchMetaTool(BaseTool):
    """
    Find a single document in a vector database by using a URL  to search the metadata field.
    """

    name: str = "Find a single document in a vector database by using a URL to search the metadata field."
    description: str = (
        "A tool that can access the metadata field of a document in a vector database."
    )
    args_schema: Type[BaseModel] = VectorMetaSearchToolSchema
    url: Optional[str] = None

    def __init__(
        self,
        elasticsearch: Elasticsearch,
        index_name: str,
        url: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._vector_database = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=BedrockEmbeddings(),
            index_name=index_name,
        )
        if url is not None:
            self.url = url
            self.description = f"A tool to find {url} in the vector database."
            self.args_schema = VectorMetaSearchToolSchema
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        url = kwargs.get("url", self.url)
        data = self._vector_database.find_document_by_url(url=url)
        # return the text of the first document
        return get_content_and_source_from_response(data)


class SerpSearchEngineIngestTool(BaseTool):
    name: str = "Search engine ingest tool"
    description: str = "A tool that can be used to ingest search engine query results into a vector database."
    args_schema: Type[BaseModel] = SerpSearchToolSchema
    search_query: Optional[str] = None
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
        elasticsearch: Elasticsearch,
        index_name: str,
        search_query: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._vector_database = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=BedrockEmbeddings(),
            index_name=index_name,
        )
        if search_query is not None:
            self.search_query = search_query
            self.description = f"A tool that can be used to search '{search_query}' in a search engine and store the results in a vector database."
            self.args_schema = SerpSearchToolSchema
            self._generate_description()

    def _ingest_page_content(self, url: str) -> str:
        try:
            return ingest(
                client=self._vector_database,
                url=url,
                headers=self.headers,
            )
        except Exception:
            return f"Error ingesting {url}"

    def _parallel_ingest_page_content(self, search_results: list[dict]) -> list[str]:
        urls = []
        for search_engine_result in search_results:
            if "organic_results" in search_engine_result:
                for result in search_engine_result["organic_results"]:
                    urls.append(result["link"])
        results = parallel_ingest(urls, self._vector_database, headers=self.headers)
        return "\n".join(results)

    def _ingest_search_results(self, search_results: list[dict]) -> str:
        all_results = []
        for search_engine_result in search_results:
            if "organic_results" in search_engine_result:
                for result in search_engine_result["organic_results"]:
                    ingested_document = self._ingest_page_content(result["link"])
                    all_results.append(ingested_document)
        return "\n".join(all_results)

    def _run(self, **kwargs: Any) -> Any:
        if os.getenv("SERPAPI_API_KEY") is None:
            raise ValueError("SERPAPI_API_KEY is not set in environment variables")
        search_query = kwargs.get("search_query")
        # run google search
        search = GoogleSearch(
            {
                "q": search_query,
                "hl": "en",
                "gl": "us",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
        google_results_dict = search.get_dict()
        # run bing search
        search = GoogleSearch(
            {
                "engine": "bing",
                "q": search_query,
                "cc": "US",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
        bing_results_dict = search.get_dict()
        all_results = [google_results_dict, bing_results_dict]
        # ingest search results
        return self._parallel_ingest_page_content(all_results)
