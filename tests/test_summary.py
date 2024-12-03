from conductor.flow.rag import WebDocumentRetriever
from conductor.flow.retriever import ElasticDocumentIdRMClient
from conductor.summarize.summarize import (
    summarize_document,
    summarize_document_parallel,
    generate_master_summary,
)
from conductor.summarize.models import SummarizedDocument, SummarizedDocuments
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import os


def test_summarize_document() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    query = "What is the history of Thomson Reuters Special Services?"
    documents = rag(query)
    summary = summarize_document(documents[0], query)
    assert isinstance(summary, SummarizedDocument)


def test_summarize_documents_parallel() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    query = "What is the history of Thomson Reuters Special Services?"
    documents = rag(query)
    summary = summarize_document_parallel(documents, query)
    assert isinstance(summary, list)


def test_generate_master_summary() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    query = "What is the history of Thomson Reuters Special Services?"
    documents = rag(query)
    summary = generate_master_summary(documents, query)
    assert isinstance(summary, SummarizedDocuments)
    assert isinstance(summary.summary, str)
