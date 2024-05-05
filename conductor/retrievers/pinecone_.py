from conductor.llms import claude_v2_1, openai_gpt_4, fireworks_mistral
from langchain_core.embeddings.embeddings import Embeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.chains import RetrievalQA
import os


class DiscordPineconeSelfQueryRetriever:
    """
    Discord Pinecone Retriever
    """

    def __init__(self, discord_pinecone_index: str, embeddings: Embeddings) -> None:
        self.metadata_field_info = [
            AttributeInfo(
                name="author",
                description="author of message in channel",
                type="string",
            ),
            AttributeInfo(
                name="created_at",
                description="time message was created",
                type="string",
            ),
            AttributeInfo(
                name="source",
                description="source of message",
                type="string",
            ),
            AttributeInfo(
                name="channel",
                description="channel message was sent in",
                type="string",
            ),
            AttributeInfo(
                name="id",
                description="id of message",
                type="string",
            ),
            AttributeInfo(
                name="text",
                description="content of message",
                type="string",
            ),
        ]
        self.document_content_description = "content of discord message"
        self.retriever = SelfQueryRetriever.from_llm(
            claude_v2_1,
            PineconeVectorStore.from_existing_index(
                index_name=discord_pinecone_index, embedding=embeddings
            ),
            self.metadata_field_info,
            self.document_content_description,
        )


def create_discord_pinecone_self_retriever():
    return DiscordPineconeSelfQueryRetriever(
        discord_pinecone_index=os.getenv("PINECONE_DISCORD_INDEX"),
        embeddings=BedrockEmbeddings(
            region_name="us-east-1",
        ),
    )


def create_claud_pinecone_discord_retriever():
    pinecone = PineconeVectorStore(
        index_name=os.getenv("PINECONE_DISCORD_INDEX"),
        embedding=BedrockEmbeddings(
            region_name="us-east-1",
        ),
        text_key="text",
    )
    return RetrievalQA.from_chain_type(
        llm=claude_v2_1, chain_type="stuff", retriever=pinecone.as_retriever()
    )


def create_gpt4_pinecone_discord_retriever():
    pinecone = PineconeVectorStore(
        index_name=os.getenv("PINECONE_DISCORD_INDEX"),
        embedding=BedrockEmbeddings(
            region_name="us-east-1",
        ),
        text_key="text",
    )
    return RetrievalQA.from_chain_type(
        llm=openai_gpt_4, chain_type="stuff", retriever=pinecone.as_retriever()
    )


def create_gpt4_pinecone_apollo_retriever():
    pinecone = PineconeVectorStore(
        index_name=os.getenv("PINECONE_APOLLO_INDEX"),
        embedding=BedrockEmbeddings(
            region_name="us-east-1",
        ),
        text_key="text",
    )
    return RetrievalQA.from_chain_type(
        llm=openai_gpt_4, chain_type="stuff", retriever=pinecone.as_retriever()
    )


def create_fireworks_pinecone_apollo_retriever():
    pinecone = PineconeVectorStore(
        index_name=os.getenv("PINECONE_APOLLO_INDEX"),
        embedding=BedrockEmbeddings(
            region_name="us-east-1",
        ),
        text_key="text",
    )
    return RetrievalQA.from_chain_type(
        llm=fireworks_mistral, chain_type="stuff", retriever=pinecone.as_retriever()
    )


def create_fireworks_pinecone_discord_retriever():
    pinecone = PineconeVectorStore(
        index_name=os.getenv("PINECONE_DISCORD_INDEX"),
        embedding=BedrockEmbeddings(
            region_name="us-east-1",
        ),
        text_key="text",
    )
    return RetrievalQA.from_chain_type(
        llm=fireworks_mistral, chain_type="stuff", retriever=pinecone.as_retriever()
    )
