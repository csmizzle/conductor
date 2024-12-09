from langchain_core.tools import tool
from conductor.flow.rag import WebDocumentRetriever, DocumentWithCredibility
from conductor.rag.embeddings import BedrockEmbeddings
from elasticsearch import Elasticsearch
import os
from loguru import logger


@tool
def get_document(query: str) -> DocumentWithCredibility:
    """
    Get documents for a given query
    """
    logger.info(f"Getting documents for query {query} ...")
    retriever = WebDocumentRetriever.with_elasticsearch_id_retriever(
        embeddings=BedrockEmbeddings(),
        elasticsearch=Elasticsearch([os.getenv("ELASTICSEARCH_URL")]),
        index_name=os.getenv("ELASTICSEARCH_INDEX"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        k=5,
        rerank_top_n=1,
    )
    return retriever(query)[0]
