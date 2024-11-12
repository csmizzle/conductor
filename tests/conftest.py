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
def elasticsearch_cloud_test_index():
    """Fixture to create and delete a test index in Elasticsearch."""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
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
def elasticsearch_cloud_test_research_index():
    """Fixture to create and delete a test index in Elasticsearch."""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
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


@pytest.fixture
def elasticsearch_test_image_index():
    """Fixture to create and delete a test index in Elasticsearch."""
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    test_index_name = "test_image_index"
    # Setup: Create the test index
    elasticsearch.indices.create(
        index=test_index_name, ignore=400
    )  # ignore 400 cause by index already exists
    # clear index
    elasticsearch.delete_by_query(
        index=test_index_name, body={"query": {"match_all": {}}}
    )
    yield test_index_name  # This allows the test to use the test index

    # Teardown: Delete the test index
    elasticsearch.indices.delete(
        index=test_index_name, ignore=[400, 404]
    )  # ignore errors if index does not exist


# @pytest.hookimpl(tryfirst=True, hookwrapper=True)
# def pytest_sessionfinish(session, exitstatus):
#     """Hook to run allure upload after pytest finishes."""
#     yield  # Let pytest finish the session first

#     # Define the upload command
#     allure_results_dir = "allure-results"

#     if os.path.exists(allure_results_dir):
#         try:
#             subprocess.run([
#                 "allure", "serve",
#                 "--configDirectory", allure_results_dir,
#                 allure_results_dir
#             ], check=True)
#             print("Allure report created.")
#         except subprocess.CalledProcessError as e:
#             print(f"Failed to upload Allure results: {e}")
#         except FileNotFoundError:
#             print("Allure CLI not found. Please ensure it's installed and in your PATH.")
#     else:
#         print(f"Allure results directory '{allure_results_dir}' not found.")
