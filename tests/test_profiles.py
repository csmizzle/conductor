from conductor.profiles.utils import specify_model
from conductor.profiles.generate import generate_profile, generate_profile_parallel
from conductor.profiles.factory import (
    create_extract_value_with_custom_type,
    create_value_rag_pipeline,
    run_value_rag_pipeline,
)
from conductor.profiles.outputs import recursive_model_dump
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


def test_create_extract_value_with_custom_type() -> None:
    value_type = int
    value_description = "Custom value description"
    custom_extract_value = create_extract_value_with_custom_type(
        value_type=value_type, value_description=value_description
    )
    assert custom_extract_value


def test_create_value_rag_pipeline() -> None:
    value_map = {
        "Farm": {
            "name": (str, "Farm name"),
            "location": (str, "Farm location"),
            "size": (int, "Farm size estimate in acres"),
        }
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
        "farm": {
            "name": (str, "Farm name"),
            "location": (str, "Farm location"),
            "size": (int, "Farm size estimate in acres"),
            "owner": (str, "Farm owner"),
            "revenue": (int, "Farm revenue"),
        }
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    specification = "Abma's Farm"
    results = run_value_rag_pipeline(specification=specification, pipeline=pipeline)
    assert isinstance(results, dict)
    data = recursive_model_dump(results)
    with open("tests/data/test_run_value_rag_pipeline.json", "w") as f:
        json.dump(data, f, indent=4)


def test_create_nested_pipeline() -> None:
    value_map = {
        "farm": {
            "name": (str, "Farm name"),
            "location": (str, "Farm location"),
            "size": (int, "Farm size estimate in acres"),
            "owner": (
                {
                    "name": (str, "Farm owner name"),
                    "title": (str, "Farm owner title"),
                },
                "Farm owner",
            ),
            "revenue": (int, "Farm revenue"),
        }
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    assert isinstance(pipeline, dict)


def test_run_nested_value_pipeline() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
    value_map = {
        "farm": {
            "name": (str, "Farm name"),
            "location": (str, "Farm location"),
            "size": (int, "Farm size estimate in acres"),
            "employees": (
                {
                    "title": (str, "Farm employee title"),
                    "phone": (str, "Farm employee phone"),
                    "email": (str, "Farm employee email"),
                },
                "Farm employees",
            ),
            # "owners": (
            #     {
            #         "name": (str, "Farm owner name"),
            #         "title": (str, "Farm owner title"),
            #         "phone": (str, "Farm owner phone"),
            #         "email": (str, "Farm owner email"),
            #     },
            #     "Farm owners",
            # ),
            "revenue": (int, "Farm revenue"),
        }
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    assert isinstance(pipeline, dict)
    specification = "The farm is Agriberry Annapolis CSA"
    results = run_value_rag_pipeline(specification=specification, pipeline=pipeline)
    assert isinstance(results, dict)
    data = recursive_model_dump(results)
    with open("tests/data/nested_farm_profile.json", "w") as f:
        json.dump(data, f, indent=4)


def test_value_extraction_not_available() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
    )
    dspy.configure(lm=search_lm)
    value_map = {
        "Company": {
            "name": (str, "Official name of the company"),
            "foreign_ownership": (bool, "Is the company US owned?"),
        }
    }
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    assert isinstance(pipeline, dict)
    specification = "Thomson Reuters"
    results = run_value_rag_pipeline(specification=specification, pipeline=pipeline)
    assert isinstance(results, dict)
    data = recursive_model_dump(results)
    with open("tests/data/test_value_extraction_not_available.json", "w") as f:
        json.dump(data, f, indent=4)
