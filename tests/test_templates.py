from conductor.template.generate import SchemaGenerator
from conductor.template import models
from tests.utils import load_model_from_test_data, save_model_to_test_data
import dspy
import os


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
