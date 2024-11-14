"""
Relationship extraction
"""
from conductor.graph.extraction import (
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
from conductor.flow.retriever import ElasticRMClient
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
elasticsearch = Elasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_URL")],
)
elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
retriever = ElasticRMClient(
    elasticsearch=elasticsearch,
    index_name=elasticsearch_test_index,
    embeddings=BedrockEmbeddings(),
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    k=20,
    rerank_top_n=10,
)


def test_relationship_extraction() -> None:
    extractor = RelationshipRAGExtractor(
        specification=specification,
        triple_types=triple_types,
        retriever=retriever,
    )
    relationships = extractor.extract()
    assert isinstance(relationships, list)
    assert all(isinstance(relationship, Relationship) for relationship in relationships)


def test_relationship_extraction_parallel() -> None:
    extractor = RelationshipRAGExtractor(
        specification=specification,
        triple_types=triple_types,
        retriever=retriever,
    )
    relationships = extractor.extract_parallel()
    assert isinstance(relationships, list)
    assert all(isinstance(relationship, Relationship) for relationship in relationships)


def test_create_graph() -> None:
    graph = create_graph(
        specification=specification, triple_types=triple_types, retriever=retriever
    )
    assert isinstance(graph, Graph)
    save_model_to_test_data(graph, "graph.json")


def test_create_graph_mini() -> None:
    dspy.configure(lm=mini)
    graph = create_graph(
        specification=specification, triple_types=triple_types, retriever=retriever
    )
    assert isinstance(graph, Graph)
    save_model_to_test_data(graph, "graph.json")
