"""
Implementation of the LLM services
"""
from langchain_openai.llms import OpenAI
from langchain_community.llms.bedrock import Bedrock
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")
openai_gpt_4 = OpenAI(temperature=0, max_tokens=2048)
claude_v2_1 = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2:1")
