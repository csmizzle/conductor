"""
Implementation of the LLM services
"""
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_aws import ChatBedrock
from botocore.config import Config
import boto3

# configure rate_limiter for Bedrock Clade Sonnet
rate_limiter = InMemoryRateLimiter(
    requests_per_second=8,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

bedrock_config = Config(
    retries={
        "max_attempts": 10,
        "mode": "adaptive",
    },
)
bedrock_runtime = boto3.client("bedrock-runtime", config=bedrock_config)
claude_sonnet = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 5000},
    rate_limiter=rate_limiter,
)
claude_haiku = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"max_tokens": 10000},
)
openai_gpt_4o = ChatOpenAI(temperature=0, max_tokens=4000, model="gpt-4o")
gpt_4o_mini = ChatOpenAI(temperature=0, max_tokens=4000, model="gpt-4o-mini")
