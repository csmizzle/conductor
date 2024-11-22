"""
Relationship extraction
"""
from conductor.graph.extraction import (
    CitedRelationshipWithCredibility,
    RelationshipRAGExtractor,
    create_graph,
)
from conductor.graph.models import (
    TripleType,
    Relationship,
    EntityType,
    RelationshipType,
    Graph,
)
from conductor.flow.rag import WebSearchRAG
from conductor.flow.retriever import ElasticDocumentIdRMClient
from conductor.rag.embeddings import BedrockEmbeddings
from elasticsearch import Elasticsearch
import os
import dspy
from tests.utils import save_model_to_test_data

bedrock_claude_sonnet = dspy.LM(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=3000,
    temperature=0.0,
)
mini = dspy.LM("gpt-4o-mini")
dspy.configure(lm=bedrock_claude_sonnet)
specification = "The company is Thomson Reuters Special Services"
triple_types = [
    TripleType(
        source=EntityType.COMPANY,
        relationship_type=RelationshipType.SUBSIDIARY,
        target=EntityType.COMPANY,
    ),
    TripleType(
        source=EntityType.COMPANY,
        relationship_type=RelationshipType.PARENT_COMPANY,
        target=EntityType.COMPANY,
    ),
    TripleType(
        source=EntityType.COMPANY,
        relationship_type=RelationshipType.ACQUIRED,
        target=EntityType.COMPANY,
    ),
    TripleType(
        source=EntityType.COMPANY,
        relationship_type=RelationshipType.EMPLOYEE,
        target=EntityType.PERSON,
    ),
]


def test_relationship_extraction() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
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
    extractor = RelationshipRAGExtractor(
        specification=specification,
        triple_types=[
            triple_types[3]
        ],  # just extract employee relationships for speed of testing ... and cost
        rag=rag,
    )
    relationships = extractor.extract()
    assert isinstance(relationships, list)
    assert all(
        isinstance(relationship, CitedRelationshipWithCredibility)
        for relationship in relationships
    )
    save_model_to_test_data(relationships, "test_relationship_extraction.json")


def test_relationship_extraction_parallel() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
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
    extractor = RelationshipRAGExtractor(
        specification=specification,
        triple_types=[
            triple_types[3]
        ],  # just extract employee relationships for speed of testing ... and cost
        rag=rag,
    )
    relationships = extractor.extract_parallel()
    assert isinstance(relationships, list)
    assert all(isinstance(relationship, Relationship) for relationship in relationships)


def test_create_graph() -> None:
    # search_lm = dspy.LM(
    #     "openai/gpt-4o",
    #     api_base=os.getenv("LITELLM_HOST"),
    #     api_key=os.getenv("LITELLM_API_KEY"),
    #     cache=False,
    #     max_tokens=3000,
    # )
    # dspy.configure(lm=search_lm)
    # elasticsearch = Elasticsearch(
    #     hosts=[os.getenv("ELASTICSEARCH_URL")],
    # )
    # elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    # retriever = ElasticDocumentIdRMClient(
    #     elasticsearch=elasticsearch,
    #     index_name=elasticsearch_test_index,
    #     embeddings=BedrockEmbeddings(),
    # )
    # rag = WebSearchRAG(elastic_id_retriever=retriever)
    # extractor = RelationshipRAGExtractor(
    #     specification=specification,
    #     triple_types=[
    #         triple_types[3]
    #     ],  # just extract employee relationships for speed of testing ... and cost
    #     rag=rag,
    # )
    # assert isinstance(graph, Graph)
    # save_model_to_test_data(graph, "graph.json")
    pass


def test_create_graph_mini() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
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
    graph = create_graph(
        specification=specification, triple_types=triple_types, rag=rag
    )
    assert isinstance(graph, Graph)
    save_model_to_test_data(graph, "graph.json")
