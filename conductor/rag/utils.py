"""
Utilities for RAG pipeline
"""
from langchain_core.documents import Document
from elastic_transport import ObjectApiResponse


def get_page_content_with_source_url(document: Document) -> str:
    """
    Get page content with source URL
    """
    return f"Source Link: {document.metadata['url']}\nContent: {document.page_content}"


def get_content_and_source_from_response(response: ObjectApiResponse) -> str:
    """
    Get content and source from response
    """
    source_document = response["hits"]["hits"][0]["_source"]
    text = source_document["text"]
    source_url = source_document["metadata"]["url"]
    return f"Source Link: {source_url}\nContent: {text}"
