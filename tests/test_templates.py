from conductor.template.generate import generate_template_schema, SchemaGenerator
import dspy
import os
import json


def test_generate_template_schema() -> None:
    search_lm = dspy.LM(
        "openai/bedrock/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        max_tokens=3000,
        cache=False,
    )
    dspy.configure(lm=search_lm)
    prompt = "I am conduct due diligence on a company."
    response = generate_template_schema(prompt)
    assert isinstance(response, dspy.Prediction)
    assert len(response.generated_fields) > 0


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
    value_map = schema_generator.generate(generate_enums=True)
    assert isinstance(value_map, dict)
    assert len(value_map) > 0


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
    assert isinstance(value_map, dict)
    assert len(value_map) > 0
    with open(
        "tests/data/test_schema_generator_with_enums_relationship_claude.json", "w"
    ) as f:
        json.dump(value_map, f, indent=4)


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
    assert isinstance(value_map, dict)
    assert len(value_map) > 0
    with open(
        "tests/data/test_schema_generator_with_enums_relationship_4o.json", "w"
    ) as f:
        json.dump(value_map, f, indent=4)


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
    assert isinstance(value_map, dict)
    assert len(value_map) > 0
    with open(
        "tests/data/test_schema_generator_with_enums_relationship_mini.json", "w"
    ) as f:
        json.dump(value_map, f, indent=4)
