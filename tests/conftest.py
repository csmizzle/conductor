from elasticsearch import Elasticsearch
import pytest
import os


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


@pytest.fixture
def elasticsearch_test_agent_index():
    """Fixture to create and delete a test index in Elasticsearch."""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    test_index_name = "test_agent_index"
    # Setup: Create the test index
    elasticsearch.indices.create(
        index=test_index_name, ignore=400
    )  # ignore 400 cause by index already exists

    yield test_index_name  # This allows the test to use the test index

    # Teardown: Delete the test index
    elasticsearch.indices.delete(
        index=test_index_name, ignore=[400, 404]
    )  # ignore errors if index does not exist
