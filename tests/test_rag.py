from conductor.flow.rag import (
    CitationRAG,
    CitedAnswerWithCredibility,
    CitedValueWithCredibility,
    AgenticCitationRAG,
    AgenticCitationValueRAG,
    WebSearchRAG,
    WebSearchValueRAG,
    get_answer,
    get_answers,
)
from conductor.flow.retriever import ElasticRMClient, ElasticDocumentIdRMClient
import os
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import dspy

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


def test_get_answer(elasticsearch_cloud_test_research_index: str) -> None:
    search_lm = dspy.LM(
        "openai/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        # cache=False,
    )
    dspy.configure(lm=search_lm)
    cloud_elastic = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
    )
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=cloud_elastic,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        k=10,
        rerank_top_n=5,
    )
    rag = WebSearchRAG(
        elastic_id_retriever=retriever,
    )
    answer = get_answer(
        rag=rag, question="Who the head of R&D and Data Science at TRSS?"
    )
    assert isinstance(answer, CitedAnswerWithCredibility)


def test_get_answers(elasticsearch_cloud_test_research_index: str) -> None:
    search_lm = dspy.LM(
        "openai/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    dspy.configure(lm=search_lm)
    cloud_elastic = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
    )
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=cloud_elastic,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        k=10,
        rerank_top_n=5,
    )
    rag = WebSearchRAG(
        elastic_id_retriever=retriever,
    )
    answer = get_answers(
        rag=rag,
        questions=[
            "Who the head of R&D and Data Science at TRSS?",
            "What is the revenue of TRSS?",
        ],
    )
    assert isinstance(answer, list)
