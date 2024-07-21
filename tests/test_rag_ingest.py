"""
Test the rag ingest components
"""
from conductor.rag.ingest import ingest_webpage
from conductor.rag.models import WebPage
import pytest


def test_ingest_webpage() -> None:
    """
    Test ingestion of a webpage
    """
    test_url = "https://flashpoint.io"
    webpage = ingest_webpage(url=test_url)
    assert isinstance(webpage, WebPage)


def test_ingest_webpage_invalid_url() -> None:
    """
    Test ingestion of a webpage with an invalid URL
    """
    test_url = "https://flashpoint.io/asdfasdf"
    with pytest.raises(Exception):
        ingest_webpage(url=test_url)
