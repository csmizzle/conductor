"""
Test the RAG client
"""
from elasticsearch import Elasticsearch
from conductor.rag.client import ElasticsearchRetrieverClient
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
    test_index_name = "test_index"
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
        title="Example",
        content="Hello, world!",
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
            title="Example",
            content="Hello, world!",
        ),
        WebPage(
            url="https://www.example.com",
            created_at=datetime.now(),
            title="Example",
            content="Hello, world!",
        ),
    ] * 10
    # create and assert writing working
    client.create_insert_webpage_documents(sample_documents)
    assert client.elasticsearch.count()["count"] == 20
    # run similarity search and assert working
    results = client.store.similarity_search(query="Hello, world!", k=1)
    assert isinstance(results, list)
    assert len(results) == 1
