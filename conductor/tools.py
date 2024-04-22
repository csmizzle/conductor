"""
tools for querying knowledge bases
"""
from langchain_core.tools import tool
from conductor.retrievers.pinecone_ import (
    create_gpt4_pinecone_apollo_retriever,
    create_gpt4_pinecone_discord_retriever,
)
from langchain.pydantic_v1 import BaseModel


class PineconeQuery(BaseModel):
    query: str


@tool("apollo-pinecone-gpt4-query", args_schema=PineconeQuery)
def apollo_pinecone_gpt4(query: str):
    """
    A Pinecone vector database with external customer data
    """
    apollo = create_gpt4_pinecone_apollo_retriever()
    return apollo.run(query)


@tool("discord-pinecone-gpt4-query", args_schema=PineconeQuery)
def discord_pinecone_gpt4(query: str):
    """
    A Pinecone vector database with internal discord data
    """
    discord = create_gpt4_pinecone_discord_retriever()
    return discord.run(query)
