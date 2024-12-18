from conductor.template.generate import SchemaGenerator
from conductor.template import models
from conductor.profiles.factory import (
    create_value_rag_pipeline,
    run_value_rag_pipeline_parallel,
)
from conductor.profiles.outputs import recursive_model_dump
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from tests.utils import load_model_from_test_data, save_model_to_test_data
import dspy
import os
import json


def test_schema_generator_with_enums() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conducting due diligence on a company."
    schema_generator = SchemaGenerator(prompt)
    value_map = schema_generator.generate()
    assert isinstance(value_map, models.ValueMap)
    save_model_to_test_data(value_map, "test_schema_generator_with_enums.json")


def test_schema_generator_with_enums_research_questions() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conducting due diligence on a company."
    schema_generator = SchemaGenerator(prompt, n_research_questions=3)
    value_map = schema_generator.generate()
    assert isinstance(value_map, models.ValueMap)
    save_model_to_test_data(
        value_map, "test_schema_generator_with_enums_research_questions.json"
    )


def test_schema_generator_with_enums_relationship_claude() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conducting due diligence on a company."
    schema_generator = SchemaGenerator(
        prompt=prompt,
        generate_nested_enums=True,
        generate_nested_relationship=True,
    )
    value_map = schema_generator.generate()
    assert isinstance(value_map, models.ValueMap)
    save_model_to_test_data(
        value_map, "test_schema_generator_with_enums_relationship_claude.json"
    )


def test_schema_generator_with_enums_relationship_4o() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conducting due diligence on a company."
    schema_generator = SchemaGenerator(
        prompt=prompt,
        generate_nested_enums=True,
        generate_nested_relationship=True,
    )
    value_map = schema_generator.generate()
    assert isinstance(value_map, models.ValueMap)
    save_model_to_test_data(
        value_map, "test_schema_generator_with_enums_relationship_4o.json"
    )


def test_load_model_from_test_data() -> None:
    model = load_model_from_test_data(
        model=models.ValueMap,
        filename="test_schema_generator_with_enums_relationship_claude.json",
    )
    assert isinstance(model, models.ValueMap)


def test_schema_generator_with_enums_relationship_mini() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conducting due diligence on a company."
    schema_generator = SchemaGenerator(
        prompt=prompt,
        generate_nested_enums=True,
        generate_nested_relationship=True,
    )
    value_map = schema_generator.generate()
    assert isinstance(value_map, models.ValueMap)
    save_model_to_test_data(
        value_map, "test_schema_generator_with_enums_relationship_mini.json"
    )


def test_run_generated_schema() -> None:
    schema = load_model_from_test_data(
        model=models.ValueMap,
        filename="test_schema_generator_with_enums_relationship_claude.json",
    )
    value_map = schema.to_value_map()
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        add_not_available=True,
    )
    assert isinstance(pipeline, dict)


def test_run_generated_schema_mini() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    schema = load_model_from_test_data(
        model=models.ValueMap,
        filename="test_schema_generator_with_enums_relationship_4o.json",
    )
    value_map = schema.to_value_map()
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        add_not_available=True,
    )
    assert isinstance(pipeline, dict)
    results = run_value_rag_pipeline_parallel(
        specification="Thomson Reuters", pipeline=pipeline, max_workers=8
    )
    assert isinstance(results, dict)
    with open("tests/data/test_run_generated_schema_mini.json", "w") as file_:
        json.dump(recursive_model_dump(results), file_, indent=4)


def test_run_generated_schema_4o() -> None:
    search_lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    schema = load_model_from_test_data(
        model=models.ValueMap,
        filename="test_schema_generator_with_enums_relationship_4o.json",
    )
    value_map = schema.to_value_map()
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        add_not_available=True,
    )
    assert isinstance(pipeline, dict)
    results = run_value_rag_pipeline_parallel(
        specification="Thomson Reuters", pipeline=pipeline, max_workers=8
    )
    assert isinstance(results, dict)
    with open("tests/data/test_run_generated_schema_4o.json", "w") as file_:
        json.dump(recursive_model_dump(results), file_, indent=4)


def test_pitcher_generated_schema_sonnet() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am researching MLB pitcher performance."
    schema_generator = SchemaGenerator(prompt, generate_nested_enums=True)
    schema = schema_generator.generate()
    value_map = schema.to_value_map()
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        add_not_available=True,
    )
    results = run_value_rag_pipeline_parallel(
        specification="David Bednar, Pittsburgh Pirates",
        pipeline=pipeline,
        max_workers=8,
    )
    with open("tests/data/test_pitcher_generated_schema_sonnet.json", "w") as file_:
        json.dump(recursive_model_dump(results), file_, indent=4)


def test_aircraft_generated_schema_claude() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-haiku",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am doing forward looking supply chain analysis of military aircraft."
    schema_generator = SchemaGenerator(prompt, generate_nested_enums=True)
    schema = schema_generator.generate()
    value_map = schema.to_value_map(include_relationships=False)
    pipeline = create_value_rag_pipeline(
        value_map=value_map,
        elasticsearch=Elasticsearch(hosts=[os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_TEST_RAG_INDEX"),
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    results = run_value_rag_pipeline_parallel(
        specification="AC130 Gunship",
        pipeline=pipeline,
        max_workers=4,
    )
    with open("tests/data/test_aircraft_generated_schema_claude.json", "w") as file_:
        json.dump(recursive_model_dump(results), file_, indent=4)
