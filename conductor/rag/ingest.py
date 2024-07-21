"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage
from unstructured.partition.text import partition_text
from unstructured.staging.base import convert_to_dict
from bs4 import BeautifulSoup
import requests
from datetime import datetime


def ingest_webpage(url: str, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
    # get a created at timestamp
    created_at = datetime.now()
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
        text = soup.get_text()
        # use the partition_text function
        elements = partition_text(text=text)
        # convert to dict
        content_dict = convert_to_dict(elements)
        # return WebPage
        return WebPage(
            url=url, created_at=created_at, content=content_dict, raw=response_text
        )
