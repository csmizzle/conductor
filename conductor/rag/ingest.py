"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage
from bs4 import BeautifulSoup
from datetime import datetime
from conductor.rag.client import ElasticsearchRetrieverClient
from zenrows import ZenRowsClient
import os


def ingest_webpage(url: str, limit: int = 50000, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
    try:
        # get a created at timestamp
        created_at = datetime.now()
        # handle pdfs by passing for now
        if not url.endswith("pdf"):
            # use zenrows to get the text from the webpage
            client = ZenRowsClient(
                apikey=os.getenv("ZENROWS_API_KEY"), retries=5, concurrency=1
            )
            if "linkedin" in url:
                params = dict(
                    js_render="true",
                    premium_proxy="true",
                )
            else:
                params = dict(
                    js_render="true",
                )
            response = client.get(
                url,
                params=params,
            )
            # process response
            if not response.ok:
                print(f"Error: {response.status_code}")
                print(f"Error: {response.text}")
                response.raise_for_status()
            else:
                # get text from response
                response_text = response.text
                # parse with BeautifulSoup
                soup = BeautifulSoup(response_text, "html.parser")
                # get text from soup
                text = soup.get_text(strip=True)
                # get the first limit characters
                text = text[:limit]
                # use the partition_text function
                return WebPage(
                    url=url, created_at=created_at, content=text, raw=response_text
                )
    except Exception as e:
        raise e


def url_to_db(url: str, client: ElasticsearchRetrieverClient, **kwargs) -> list[str]:
    """
    Ingest webpage from URL to Elasticsearch
    """
    # ingest webpage
    webpage = ingest_webpage(url, **kwargs)
    # insert document
    return client.create_insert_webpage_document(webpage)
