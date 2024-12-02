from conductor.profiles.factory import (
    GraphModelFactoryPipeline,
    GraphSignatureFactory,
    RelationshipRAGExtractorFactory,
)
from conductor.graph.create import create_deduplicated_graph
from elasticsearch import Elasticsearch
from conductor.flow.credibility import SourceCredibility
from conductor.flow.rag import WebDocumentRetriever
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.flow.retriever import ElasticDocumentIdRMClient
from tests.utils import save_model_to_test_data
import os
import dspy


def test_graph_model_factory_pipeline():
    triple_types = {
        "SUBSIDIARY": ("COMPANY", "COMPANY"),
        "EMPLOYEE": ("PERSON", "COMPANY"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    assert pipeline.relationship_type is not None
    pipeline.create_triple_type_input()
    assert pipeline.triple_type_inputs is not None


def test_graph_signatures_factory():
    triple_types = {
        "SUBSIDIARY": ("COMPANY", "COMPANY"),
        "EMPLOYEE": ("PERSON", "COMPANY"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    pipeline.create_triple_type_input()
    factory = GraphSignatureFactory(
        triple_type_model=pipeline.created_triple_type_model,
        relationship=pipeline.relationship_model,
    )
    factory.create_signatures()
    assert factory.relationship_query is not None
    assert factory.extracted_relationships is not None
    assert factory.relationship_reasoning is not None
    map_ = factory.create_signatures_map()
    assert isinstance(map_, dict)


def test_graph_create_pipeline_functions() -> None:
    triple_types = {
        "FOLLOWER": ("USER", "USER"),
        "SUBSCRIBER": ("USER", "INFLUENCER"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline._create_relationship_types()
    # ensure the relationship model is created
    assert pipeline.relationship_type is not None
    # create the entity types
    pipeline._create_entity_types()
    assert pipeline.entity_type is not None
    # create the entity model
    pipeline._create_entity_base_model()
    # use the entity model to create a new instance
    entity_model = pipeline.entity_model
    entity_instance = entity_model(
        entity_type="USER",
        name="Adam",
    )
    assert entity_instance.entity_type == "USER"
    # create the realtionship models
    pipeline._create_relationship_base_model()
    # create the relationship model
    relationship_model = pipeline.relationship_model
    relationship = relationship_model(
        source=entity_instance,
        target=entity_instance,
        relationship_type="FOLLOWER",
        faithfulness=5,
        factual_correctness=5,
        confidence=5,
    )
    # create the cited relationship models for the pipeline
    pipeline._create_cited_relationship()
    # create the cited relationship model
    relationship_model = pipeline.created_cited_relationship
    relationship = relationship_model(
        source=entity_instance,
        target=entity_instance,
        relationship_type="FOLLOWER",
        relationships_query="Acme Inc. is a subsidiary of Acme Inc.",
        relationship_faithfulness=5,
        relationship_factual_correctness=5,
        relationship_reasoning="The relationship is well documented.",
        relationship_confidence=5,
        document_content="Acme Inc. is a subsidiary of Acme Inc.",
        document_source="https://example.com",
        document_source_credibility=SourceCredibility(
            source="https://example.com",
            credibility="HIGH",
        ),
        document_source_credibility_reasoning="The source is a reliable news outlet.",
    )
    assert relationship.relationship_type == "FOLLOWER"
    # create triple type inputs
    pipeline._create_triple_type_model()
    # create the triple type input model
    triple_type_model = pipeline.created_triple_type_model
    triple_type = triple_type_model(
        source="USER",
        target="USER",
        relationship_type="FOLLOWER",
    )
    assert triple_type.relationship_type == "FOLLOWER"


def test_relationship_factory_extract() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
    triple_types = {
        "SUBSIDIARY": ("COMPANY", "COMPANY"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    pipeline.create_triple_type_input()
    factory = GraphSignatureFactory(
        triple_type_model=pipeline.created_triple_type_model,
        relationship=pipeline.relationship_model,
    )
    factory.create_signatures()
    map_ = factory.create_signatures_map()
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
    rag_factory = RelationshipRAGExtractorFactory(
        specification="The company is Thomson Reuters.",
        triple_types=pipeline.triple_type_inputs,
        rag_instance=rag,
        signature_map=map_,
        cited_relationship_model=pipeline.created_cited_relationship,
    )
    rag_factory.create_instance()
    assert rag_factory.created_rag_instance is not None
    custom_rag = rag_factory.created_rag_instance
    relationships = custom_rag.extract()
    assert relationships is not None


def test_relationship_factory_extract_parallel() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
    triple_types = {
        # "EXECUTIVE": ("COMPANY", "PERSON"),
        # "SUBSIDIARY": ("COMPANY", "COMPANY"),
        "LOCATIONS": ("COMPANY", "LOCATION"),
    }
    pipeline = GraphModelFactoryPipeline(triple_types)
    pipeline.create_models()
    pipeline.create_triple_type_input()
    factory = GraphSignatureFactory(
        triple_type_model=pipeline.created_triple_type_model,
        relationship=pipeline.relationship_model,
    )
    factory.create_signatures()
    map_ = factory.create_signatures_map()
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
    rag_factory = RelationshipRAGExtractorFactory(
        specification="The company is Thomson Reuters.",
        triple_types=pipeline.triple_type_inputs,
        rag_instance=rag,
        signature_map=map_,
        cited_relationship_model=pipeline.created_cited_relationship,
    )
    rag_factory.create_instance()
    assert rag_factory.created_rag_instance is not None
    custom_rag = rag_factory.created_rag_instance
    relationships = custom_rag.extract_parallel()
    assert relationships is not None
    graph = create_deduplicated_graph(relationships)
    save_model_to_test_data(
        graph, "test_unique_relationship_factory_extract_parallel.json"
    )
