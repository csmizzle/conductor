"""
Test the rag ingest components
"""
from conductor.rag.ingest import (
    ingest_webpage,
    ingest_with_ids,
    parallel_ingest_with_ids,
)
from conductor.rag.models import WebPage
from conductor.rag.client import ElasticsearchRetrieverClient, Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import pytest
import os


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


def test_ingest_with_ids() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
    )
    test_url = "https://trssllc.com"
    webpages = ingest_with_ids(client=client, url=test_url, size=None)
    assert isinstance(webpages, dict)
    # retrieve chunks by document id
    documents = client.mget_documents(document_ids=webpages[test_url])
    assert isinstance(documents, list)


def test_parrell_ingest_with_ids() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
    )
    test_urls = ["https://trssllc.com", "https://flashpoint.io"]
    webpages = parallel_ingest_with_ids(client=client, urls=test_urls, size=None)
    assert isinstance(webpages, dict)
    # retrieve chunks by document id
    ids = []
    for url in test_urls:
        ids.extend(webpages[url])
    documents = client.mget_documents(document_ids=ids)
    assert isinstance(documents, list)


def test_ingest_webpage_pdf() -> None:
    # url = "http://senecacountycce.org/resources/guide-to-marketing-channels"
    url = "https://arxiv.org/pdf/2408.09869"
    webpage = ingest_webpage(url=url)
    assert isinstance(webpage, WebPage)


def test_parallel_ingest_pdf_with_ids() -> None:
    url = "https://arxiv.org/pdf/2408.09869"
    test_urls = [url]
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
    )
    webpages = parallel_ingest_with_ids(client=client, urls=test_urls)
    assert isinstance(webpages, dict)
    # retrieve chunks by document id
    ids = []
    for url in test_urls:
        ids.extend(webpages[url])
    documents = client.mget_documents(document_ids=ids)
    assert isinstance(documents, list)
    assert len(documents) == 12
