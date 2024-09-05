"""
Test the chains module.
"""
from conductor.chains import (
    run_set_graph_chain,
    Graph,
    Timeline,
    QueryMatch,
    RelationshipType,
    SyntheticDocuments,
    run_timeline_chain,
    run_query_match_chain,
    match_queries_to_paragraphs,
    run_create_caption_chain,
    run_hyde_generation_chain,
    run_hyde_search,
    run_sourced_section_chain,
)
from conductor.rag.ingest import relationships_to_image_query, queries_to_image_results
from conductor.rag.client import ElasticsearchRetrieverClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.reports.models import ReportV2, SectionV2
from tests.constants import TEST_COMPLEX_NARRATIVE, GRAPH_JSON, REPORT_V2_JSON
import os
import json


def test_run_set_graph_chain() -> None:
    graph = run_set_graph_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(graph, Graph)
    # with open("tests/data/test_graph.json", "w") as f:
    #     json.dump(graph.model_dump(), f, indent=4)


def test_run_timeline_chain() -> None:
    timeline = run_timeline_chain(TEST_COMPLEX_NARRATIVE)
    assert isinstance(timeline, Timeline)


def test_run_query_match_chain() -> None:
    query_match = run_query_match_chain(
        search_query="John Fallone Fallone SV",
        text="""
        At the helm of Fallone SV is John Fallone, who serves as the Attorney and Managing Member. With
        over a decade of experience as a startup founder and legal counsel, John brings a wealth of knowledge
        to the firm. He co-founded SendHub, a startup that successfully raised approximately $10 million before
        its acquisition. John's academic credentials include a J.D. from George Mason Law School and an
        LL.M from Duke Law. Licensed in both California and North Carolina, he assists startups with venture
        financing, corporate structuring, contracts, mergers, and acquisitions. John is also a member of the
        """,
    )
    assert isinstance(query_match, QueryMatch)
    assert query_match.determination == "RELEVANT"


def test_match_queries_to_paragraphs() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    image_queries = relationships_to_image_query(
        graph=graph,
        relationship_types=[
            RelationshipType.FOUNDER.value,
            RelationshipType.EMPLOYEE.value,
        ],
        api_key=os.getenv("SERPAPI_API_KEY"),
    )
    assert isinstance(image_queries, list)
    image_search_results = queries_to_image_results(
        search_queries=image_queries, n_images=1
    )
    assert isinstance(image_search_results, list)
    assert len(image_search_results) == 7
    matched_paragraphs = match_queries_to_paragraphs(
        image_search_results=image_search_results,
        sections_filter=["Company Structure", "Personnel"],
        report=ReportV2.model_validate(REPORT_V2_JSON),
    )
    with open("tests/data/matched_paragraphs.json", "w") as f:
        json.dump(matched_paragraphs.model_dump(), f, indent=4)
    assert isinstance(matched_paragraphs, ReportV2)


def test_caption_chain() -> None:
    caption = run_create_caption_chain(
        image_title="Jim Dinkins - Thomson Reuters Institute",
        search_query="Jim Dinkins TRSS",
    )
    assert isinstance(caption, str)


def test_hyde_chain() -> None:
    synthetic_documents = run_hyde_generation_chain(
        context="Thomson Reuters Special Services",
        objective="Build out a report section on the company's leadership team",
        n_documents=5,
    )
    assert isinstance(synthetic_documents, SyntheticDocuments)


def test_hyde_search_chain() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_INDEX"),
    )
    documents = run_hyde_search(
        context="Thomson Reuters Special Services",
        objective="Build out a report section on the company's leadership team",
        retriever=client,
    )
    assert isinstance(documents, list)


def test_run_sourced_section_chain() -> None:
    sourced_section = run_sourced_section_chain(
        title="Company Leadership",
        style="NARRATIVE",
        tone="ANALYTICAL",
        point_of_view="THIRD_PERSON",
        context="Thomson Reuters Special Services is a company that provides a range of services to clients in the financial industry led by Bob Smith in washington, DC. Source: trssllc.com.",
    )
    assert isinstance(sourced_section, SectionV2)


def test_write_sourced_section() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_INDEX"),
    )
    documents = run_hyde_search(
        context="Thomson Reuters Special Services",
        objective="Build out a report section on the company's leadership team. Include all personnel and their roles. Be extremely detailed and thorough.",
        retriever=client,
        n_documents=10,
    )
    documents = [
        doc.page_content + "\nSource:" + doc.metadata["url"] for doc in documents
    ]
    sourced_section = run_sourced_section_chain(
        title="Company Leadership",
        style="as long form narratives, avoiding bullet points and short sentences.",
        tone="ANALYTICAL",
        point_of_view="THIRD_PERSON",
        context=documents,
    )
    assert isinstance(sourced_section, SectionV2)
    with open("tests/data/test_sourced_section.json", "w") as f:
        json.dump(sourced_section.model_dump(), f, indent=4)
