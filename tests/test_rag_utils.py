from conductor.rag.utils import (
    get_content_and_source_from_response,
    get_page_content_with_source_url,
)
from langchain_core.documents import Document
from datetime import datetime
from elasticsearch import Elasticsearch
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.rag.ingest import url_to_db
from conductor.rag.embeddings import BedrockEmbeddings
import os


def test_get_page_content_with_source_url():
    """
    Test the get_page_content_with_source_url function.
    """
    document = Document(
        page_content="Hello, world!",
        metadata={
            "url": "https://www.example.com",
            "created_at": datetime.now(),
            "raw": "Hello, world!",
        },
    )
    result = get_page_content_with_source_url(document)
    assert isinstance(result, str)
    assert "Source Link: https://www.example.com" in result
    assert "Content: Hello, world!" in result


def test_get_content_and_source_from_response(elasticsearch_test_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=BedrockEmbeddings(),
        index_name=elasticsearch_test_index,
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Accept-Encoding": "gzip, deflate, br",
    }
    url = "https://trssllc.com"
    url_to_db(url, client, headers=headers)
    result = client.find_webpage_by_url(url=url)
    data_with_source = get_content_and_source_from_response(result)
    assert isinstance(data_with_source, str)
