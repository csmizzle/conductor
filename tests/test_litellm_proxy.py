"""
Test out litellm local proxy
"""
import dspy


def test_bedrock_llm_dspy():
    lm = dspy.LM(
        "openai/claude-3-5-sonnet",
        api_base="http://0.0.0.0:4000",
    )
    lm("What is the meaning of life?")


def test_gpt_llm_dspy():
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base="http://0.0.0.0:4000",
    )
    lm("What is the meaning of life?")
