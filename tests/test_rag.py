from conductor.flow.rag import (
    CitationRAG,
    CitedAnswerWithCredibility,
    CitedValueWithCredibility,
    AgenticCitationRAG,
    AgenticCitationValueRAG,
    WebSearchRAG,
    WebSearchValueRAG,
)
from conductor.flow.retriever import ElasticRMClient, ElasticDocumentIdRMClient
import os
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings

query = (
    "What are TRSS main customer groups? Are there any specific agencies they serve?"
)


def test_rag() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = CitationRAG(elastic_retriever=retriever)
    answer = rag.forward(question=query)
    assert isinstance(answer, CitedAnswerWithCredibility)


def test_agentic_rag() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = AgenticCitationRAG(elastic_retriever=retriever)
    answer = rag(question="Who is the CFO of TRSS?")
    assert isinstance(answer, CitedAnswerWithCredibility)


def test_agentic_rag_value() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = AgenticCitationValueRAG(elastic_retriever=retriever)
    value = rag(question="Who is the Chief Technology Officer of TRSS?")
    assert isinstance(value, CitedValueWithCredibility)


def test_web_search_rag() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebSearchRAG(elastic_id_retriever=retriever)
    answer = rag(question="Who the head of R&D and Data Science at TRSS?")
    assert isinstance(answer, CitedAnswerWithCredibility)


def test_web_search_value_rag() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebSearchValueRAG(elastic_id_retriever=retriever)
    value = rag(question="What is the revenue of TRSS?")
    assert isinstance(value, CitedValueWithCredibility)
