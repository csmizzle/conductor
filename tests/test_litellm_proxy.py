"""
Test out litellm local proxy
"""
import dspy
import os


def test_bedrock_llm_dspy():
    lm = dspy.LM(
        "openai/claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    response = lm("What is the meaning of life today?")
    assert response


def test_gpt_llm_dspy():
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    lm("What is the meaning of life today?")


def test_llm_does_not_exist():
    lm = dspy.LM(
        "openai/this-deosnt-exist",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    lm("What is the meaning of life today?")
