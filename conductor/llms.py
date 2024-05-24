"""
Implementation of the LLM services
"""
from langchain_openai.llms import OpenAI
from langchain_aws import ChatBedrock, BedrockLLM
from langchain_fireworks.llms import Fireworks
import boto3

bedrock_runtime = boto3.client("bedrock-runtime")
openai_gpt_4 = OpenAI(temperature=0, max_tokens=2048)
claude_v2_1 = BedrockLLM(
    client=bedrock_runtime, model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)
fireworks_mistral = Fireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct", max_tokens=248
)
chat_bedrock = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.1},
)
