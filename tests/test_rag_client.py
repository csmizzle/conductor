"""
Test the RAG client
"""
from elasticsearch import Elasticsearch
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.rag.ingest import url_to_db
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.rag.models import WebPage
from datetime import datetime
import os
import pytest


@pytest.fixture
def elasticsearch_test_index():
    """Fixture to create and delete a test index in Elasticsearch."""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    test_index_name = "test_client_index"
    # Setup: Create the test index
    elasticsearch.indices.create(
        index=test_index_name, ignore=400
    )  # ignore 400 cause by index already exists

    yield test_index_name  # This allows the test to use the test index

    # Teardown: Delete the test index
    elasticsearch.indices.delete(
        index=test_index_name, ignore=[400, 404]
    )  # ignore errors if index does not exist


def test_elasticsearch_retriever_client_single_document(elasticsearch_test_index):
    """Test out the ElasticsearchRetrieverClient with sample data"""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=elasticsearch_test_index,
    )
    sample_document = WebPage(
        url="https://www.example.com",
        created_at=datetime.now(),
        content="Hello, world!",
        raw="Hello, world!",
    )
    # create and assert writing working
    client.create_insert_webpage_document(sample_document)
    assert client.elasticsearch.count()["count"] == 1
    # run similarity search and assert working
    results = client.store.similarity_search(query="Hello, world!", top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1


def test_elasticsearch_retriever_client_multiple_documents(elasticsearch_test_index):
    """Test out the ElasticsearchRetrieverClient with multiple sample data"""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=elasticsearch_test_index,
    )
    sample_documents = [
        WebPage(
            url="https://www.example.com",
            created_at=datetime.now(),
            content="Hello, world!",
            raw="Hello, world!",
        ),
        WebPage(
            url="https://www.example.com",
            created_at=datetime.now(),
            content="Hello, world!",
            raw="Hello, world!",
        ),
    ] * 10
    # create and assert writing working
    client.create_insert_webpage_documents(sample_documents)
    assert client.elasticsearch.count()["count"] == 20
    # run similarity search and assert working
    results = client.store.similarity_search(query="Hello, world!", k=1)
    assert isinstance(results, list)
    assert len(results) == 1


def test_url_to_db(elasticsearch_test_index):
    """Test the url_to_db function"""
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
    document_ids = url_to_db(url, client, headers=headers)
    assert len(document_ids) == 1
    assert client.elasticsearch.count()["count"] == 2
    # run similarity search and assert working
    results = client.store.similarity_search(
        query="Thomson Reuters Special Services", top_k=1
    )
    assert isinstance(results, list)
    assert len(results) == 1
    # test if we can find the document using the find by metadata url function
    document = client.find_webpage_by_url(url)
    assert document["hits"]["total"]["value"] == 1
