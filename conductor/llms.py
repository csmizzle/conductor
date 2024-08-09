"""
Implementation of the LLM services
"""
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models.bedrock import BedrockChat
from botocore.config import Config
import boto3


bedrock_config = Config(
    retries={
        "max_attempts": 10,
        "mode": "adaptive",
    },
)
bedrock_runtime = boto3.client("bedrock-runtime", config=bedrock_config)
openai_gpt_4 = OpenAI(temperature=0, max_tokens=2048)
claude_v2_1 = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2:1")
claude_sonnet = BedrockChat(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 5000},
)
claude_haiku = BedrockChat(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"max_tokens": 10000},
)
openai_gpt_4o = ChatOpenAI(temperature=0, max_tokens=4000, model="gpt-4o")
gpt_4o_mini = ChatOpenAI(temperature=0, max_tokens=4000, model="gpt-4o-mini")
