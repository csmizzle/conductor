from conductor.flow.retriever import ElasticRMClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import os
import dspy
import json


query = "What are TRSS products?"


def write_rag_results(documents, file_name: str) -> None:
    cleaned_documents = [{"content": document} for document in documents.documents]
    with open(file_name, "w") as f:
        json.dump(cleaned_documents, f, indent=4)


def test_flow_retrieve_with_cohere() -> None:
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    documents = retriever.forward(
        query=query,
    )
    assert isinstance(documents, dspy.Prediction)
    assert len(documents.documents) == 3
    write_rag_results(documents, "./tests/data/test_reranked_rag_results.json")


def test_flow_retrieve() -> None:
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    documents = retriever.forward(
        query=query,
    )
    assert isinstance(documents, dspy.Prediction)
    assert len(documents.documents) == 3
    write_rag_results(documents, "./tests/data/test_rag_results.json")
