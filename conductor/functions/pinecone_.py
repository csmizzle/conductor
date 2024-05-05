"""
Functions that search pinecone indexes
"""
from conductor.retrievers.pinecone_ import (
    create_fireworks_pinecone_apollo_retriever,
    create_fireworks_pinecone_discord_retriever,
)
from conductor.chains import summarize

fireworks_apollo = create_fireworks_pinecone_apollo_retriever()
fireworks_discord = create_fireworks_pinecone_discord_retriever()


def search_pinecone(query: str) -> dict:
    """
    Search query in all pinecone indexes
    """
    apollo = fireworks_apollo.invoke({"query": query})
    discord = fireworks_discord.invoke({"query": query})
    summary = summarize(apollo["result"] + "\n" + discord["result"])
    return summary
