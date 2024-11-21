from conductor.profiles.utils import specify_model
from conductor.profiles.generate import generate_profile, generate_profile_parallel
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
    specified_fields = specify_model(model=Company, specification=specification)
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


@with_langtrace_root_span(name="test_generate_profile_parallel")
def test_generate_profile_parallel():
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
    profile = generate_profile_parallel(
        model_schema=Company.model_json_schema(), specification=specification, rag=rag
    )
    assert isinstance(profile, dict)
    new_profile = {}
    for key, value in profile.items():
        new_profile[key] = value.model_dump()
    with open("tests/data/company_profile_parallel.json", "w") as f:
        json.dump(new_profile, f, indent=4)
