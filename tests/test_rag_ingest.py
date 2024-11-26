"""
Test the rag ingest components
"""
from conductor.rag.ingest import (
    ingest_webpage,
    describe_image_from_url,
    relationships_to_image_query,
    queries_to_image_results,
    describe_from_image_search_results,
    ingest_with_ids,
    parallel_ingest_with_ids,
)
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.llms import openai_gpt_4o
from tests.constants import GRAPH_JSON
from conductor.reports.models import Graph, RelationshipType
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


def test_ingest_image_from_url() -> None:
    """
    Test ingestion of an image from a URL
    """
    test_url = "https://assets.weforum.org/sf_account/image/-T6sEZZYrPjKBFqgJR9nhnbLpKoafHG__y0ZlbMJaU8.jpg"
    image = describe_image_from_url(
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
        describe_image_from_url(
            image_url=test_url,
            model=openai_gpt_4o,
            metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
        )


def test_graph_to_images() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    image_queries = relationships_to_image_query(
        graph=graph,
        relationship_types=[RelationshipType.FOUNDER.value],
        api_key=os.getenv("SERPAPI_API_KEY"),
    )
    assert isinstance(image_queries, list)
    image_search_results = queries_to_image_results(
        search_queries=image_queries, n_images=1
    )
    assert isinstance(image_search_results, list)
    assert len(image_search_results) == 2
    descriptions = describe_from_image_search_results(
        image_results=image_search_results, model=openai_gpt_4o
    )
    assert isinstance(descriptions, list)
    assert len(descriptions) == 1


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
