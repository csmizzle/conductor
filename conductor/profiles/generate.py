"""
Generate RAG based profile building
"""
from elasticsearch import Elasticsearch
from pydantic import BaseModel, InstanceOf
from langchain_core.embeddings import Embeddings
from conductor.flow.rag import CitationValueRAG
from conductor.flow.retriever import ElasticRMClient
from conductor.profiles.utils import specify_model
import concurrent.futures


def generate_profile(
    model: InstanceOf[BaseModel],
    embeddings: InstanceOf[Embeddings],
    specification: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    cohere_api_key: str = None,
) -> BaseModel:
    """
    Generate a profile based on a model and a specification
    """
    specified_fields = specify_model(model=model, specification=specification)
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=index_name,
        embeddings=embeddings,
        cohere_api_key=cohere_api_key,
    )
    rag = CitationValueRAG(elastic_retriever=retriever)
    profile = {}
    for field in specified_fields:
        profile[field] = rag(question=specified_fields[field])
    created_profile = model(**profile)
    return created_profile


def generate_profile_parallel(
    model: InstanceOf[BaseModel],
    embeddings: InstanceOf[Embeddings],
    specification: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    cohere_api_key: str = None,
) -> BaseModel:
    """
    Generate a profile based on a model and a specification
    """
    specified_fields = specify_model(model=model, specification=specification)
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=index_name,
        embeddings=embeddings,
        cohere_api_key=cohere_api_key,
    )
    rag = CitationValueRAG(elastic_retriever=retriever)
    profile = {}
    # Use concurrent futures to parallelize the profile generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(rag, question=specified_fields[field]): field
            for field in specified_fields
        }
        for future in concurrent.futures.as_completed(futures):
            field = futures[future]
            profile[field] = future.result()
    created_profile = model(**profile)
    return created_profile
