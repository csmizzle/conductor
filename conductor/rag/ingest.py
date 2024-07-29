"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from conductor.rag.client import ElasticsearchRetrieverClient


def ingest_webpage(url: str, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
    # get a created at timestamp
    created_at = datetime.now()
    # handle pdfs by passing for now
    if not url.endswith("pdf"):
        # use requests
        response = requests.get(url, **kwargs)
        # process response
        if not response.ok:
            response.raise_for_status()
        else:
            # get text from response
            response_text = response.text
            # parse with BeautifulSoup
            soup = BeautifulSoup(response_text, "html.parser")
            # get text from soup
            text = soup.get_text(strip=True)
            # use the partition_text function
            return WebPage(
                url=url, created_at=created_at, content=text, raw=response_text
            )
    else:
        return WebPage(
            url=url, content="Unable to parse PDF", raw="", created_at=created_at
        )


def url_to_db(url: str, client: ElasticsearchRetrieverClient, **kwargs) -> list[str]:
    """
    Ingest webpage from URL to Elasticsearch
    """
    # ingest webpage
    webpage = ingest_webpage(url, **kwargs)
    # insert document
    return client.create_insert_webpage_document(webpage)
