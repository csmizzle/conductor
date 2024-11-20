from conductor.flow.rag import (
    CitationRAG,
    CitedAnswerWithCredibility,
    AgenticCitationRAG,
)
from conductor.flow.retriever import ElasticRMClient
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
