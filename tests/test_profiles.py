from conductor.profiles.utils import specify_model
from conductor.profiles.generate import generate_profile, generate_profile_parallel
from conductor.profiles.factory import (
    create_custom_cited_value,
    create_custom_cited_model,
    create_extract_value_with_custom_type,
    create_web_search_value_rag,
    create_value_rag_pipeline,
    run_value_rag_pipeline,
)
from conductor.profiles.models import Company
from conductor.flow.rag import WebSearchValueRAG
from conductor.rag.embeddings import BedrockEmbeddings
from elasticsearch import Elasticsearch
import dspy
import os
import json
from langtrace_python_sdk import with_langtrace_root_span


# configure dspy
llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)


def test_specify_model():
    specification = "The company is Thomson Reuters"
    specified_fields = specify_model(
        model_schema=Company.model_json_schema(), specification=specification
    )
    assert specified_fields


def test_generate_profile():
    specification = "The company is Thomson Reuters Special Services"
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    rag = WebSearchValueRAG.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    profile = generate_profile(
        model_schema=Company.model_json_schema(), specification=specification, rag=rag
    )
    assert isinstance(profile, dict)
    new_profile = {}
    for key, value in profile.items():
        new_profile[key] = value.model_dump()
    with open("tests/data/company_profile.json", "w") as f:
        json.dump(new_profile, f, indent=4)
    with open("tests/data/abstract_company_profile.json", "w") as f:
        json.dump(Company.model_json_schema(), f, indent=4)


@with_langtrace_root_span(name="test_generate_profile_parallel")
def test_generate_profile_parallel():
    specification = "The farm is Abma's Farm"
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    rag = WebSearchValueRAG.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    profile = generate_profile_parallel(
        model_schema=Company.model_json_schema(), specification=specification, rag=rag
    )
    assert isinstance(profile, dict)
    new_profile = {}
    for key, value in profile.items():
        new_profile[key] = value.model_dump()
    with open("tests/data/farm_profile_parallel.json", "w") as f:
        json.dump(new_profile, f, indent=4)


def test_create_custom_cited_value() -> None:
    value_type = str
    value_description = "Custom value description"
    custom_cited_value = create_custom_cited_value(
        value_type=value_type, value_description=value_description
    )
    assert custom_cited_value


def test_create_custom_cited_model() -> None:
    model_name = "Farm"
    value_map = {
        "name": (str, "Farm name"),
        "location": (str, "Farm location"),
        "size": (int, "Farm size estimate in acres"),
    }
    custom_cited_model = create_custom_cited_model(
        model_name=model_name, value_map=value_map
    )
    assert custom_cited_model
    with open("tests/data/abstract_farm_model.json", "w") as f:
        json.dump(custom_cited_model.model_json_schema(), f, indent=4)


def test_specify_custom_model() -> None:
    model_name = "Farm"
    value_map = {
        "name": (str, "Farm name"),
        "location": (str, "Farm location"),
        "size": (int, "Farm size estimate in acres"),
    }
    custom_cited_model = create_custom_cited_model(
        model_name=model_name, value_map=value_map
    )
    # specify the model
    specification = "The farm is Abma's Farm"
    specified_fields = specify_model(
        model_schema=custom_cited_model.model_json_schema(), specification=specification
    )
    assert isinstance(specified_fields, dict)


def test_create_extract_value_with_custom_type() -> None:
    value_type = int
    value_description = "Custom value description"
    custom_extract_value = create_extract_value_with_custom_type(
        value_type=value_type, value_description=value_description
    )
    assert custom_extract_value


def test_create_custom_rag() -> None:
    value_type = int
    value_description = "Company revenue"
    custom_cited_value = create_custom_cited_value(
        value_type=value_type, value_description=value_description
    )
    custom_extract_value = create_extract_value_with_custom_type(
        value_type=value_type, value_description=value_description
    )
    rag = create_web_search_value_rag(
        extract_value=custom_extract_value, return_class=custom_cited_value
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    created_rag = rag.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    answer = created_rag(question="What is the revenue of TRSS?")
    assert isinstance(answer.value, int)


def test_create_value_rag_pipeline() -> None:
    value_map = {
        "name": (str, "Farm name"),
        "location": (str, "Farm location"),
        "size": (int, "Farm size estimate in acres"),
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    assert isinstance(pipeline, dict)


def test_run_value_rag_pipeline() -> None:
    value_map = {
        "name": (str, "Farm name"),
        "location": (str, "Farm location"),
        "size": (int, "Farm size estimate in acres"),
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    specification = "The farm is Abma's Farm"
    results = run_value_rag_pipeline(specification=specification, pipeline=pipeline)
    assert isinstance(results, dict)
