"""
Test the rag ingest components
"""
from conductor.rag.ingest import ingest_webpage, ingest_image_from_url
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.llms import openai_gpt_4o
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


def test_ingest_image_from_url() -> None:
    """
    Test ingestion of an image from a URL
    """
    test_url = "https://assets.weforum.org/sf_account/image/-T6sEZZYrPjKBFqgJR9nhnbLpKoafHG__y0ZlbMJaU8.jpg"
    image = ingest_image_from_url(
        image_url=test_url,
        model=openai_gpt_4o,
        metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
    )
    assert isinstance(image, SourcedImageDescription)


def test_ingest_image_from_url_invalid_url() -> None:
    """
    Test ingestion of an image from an invalid URL
    """
    test_url = "https://flashpoint.io/asdfasdf"
    with pytest.raises(Exception):
        ingest_image_from_url(
            image_url=test_url,
            model=openai_gpt_4o,
            metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
        )
