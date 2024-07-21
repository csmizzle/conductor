"""
Add two different kinds of tools
- Scrape into a vector database
- Search a vector database
"""
from crewai_tools.tools import ScrapeWebsiteTool
from crewai_tools.tools.base_tool import BaseTool
from pydantic.v1 import BaseModel, Field
from typing import Optional, Any, Type, List
from langchain_core.documents import Document
from conductor.crews.marketing.tools import (
    ScrapeWebsiteToolSchema,
    SerpSearchToolSchema,
)
from conductor.rag.ingest import url_to_db
from conductor.rag.client import ElasticsearchRetrieverClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from serpapi import GoogleSearch
import os


class FixedVectorSearchToolSchema(BaseModel):
    """Input for SerpSearchTool."""

    pass


class VectorSearchToolSchema(FixedVectorSearchToolSchema):
    """Input for SerpSearchTool."""

    search_query: str = Field(
        ...,
        description="A question that will be used to search the vector database.",
    )


def ingest(
    client: ElasticsearchRetrieverClient,
    url: str,
    headers: dict = None,
    cookies: dict = None,
):
    print(f"Ingesting data for {url} ...")
    existing_document = client.find_webpage_by_url(url=url)
    if existing_document["hits"]["total"]["value"] > 0:
        return "Document already exists in the vector database"
    else:
        webpage = url_to_db(url=url, client=client, headers=headers, cookies=cookies)
        return f"New documents added: {', '.join(webpage)}"


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
        documents: List[Document] = self._vector_database.store.similarity_search(
            query=search_query, k=1
        )
        return "\n".join([document.page_content for document in documents])


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
        # add organic results to search context
        if "organic_results" in results_dict:
            for result in results_dict["organic_results"]:
                ingested_document = self._ingest_page_content(result["link"])
                search_context.append(ingested_document)
        return "\n".join(search_context)
