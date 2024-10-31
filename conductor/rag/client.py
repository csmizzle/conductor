"""
Ingestion logic for RAG data
- Ingest website data
- Clean data
- Vectorize data
- Store data
"""
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from conductor.rag.models import WebPage, SourcedImageDescription


class ElasticsearchRetrieverClient:
    """
    Ingest documents into Elasticsearch
    """

    def __init__(
        self,
        elasticsearch: Elasticsearch,
        embeddings: Embeddings,
        index_name: str,
    ) -> None:
        self.elasticsearch = elasticsearch
        self.embeddings = embeddings
        self.store = ElasticsearchStore(
            index_name=index_name,
            es_connection=elasticsearch,
            embedding=embeddings,
        )
        self.index_name = index_name

    def create_image_document(self, image: SourcedImageDescription) -> Document:
        return Document(
            page_content=image.image_description.combine_description_metadata(),
            metadata={
                "url": image.source,
                "created_at": image.created_at,
                "path": image.path,
                "image_metadata": image.image_description.metadata,
                "description": image.image_description.description,
            },
        )

    def create_webpage_document(self, webpage: WebPage) -> Document:
        """
        Ingest webpage document into Elasticsearch
        """
        return Document(
            page_content=webpage.content,
            metadata={
                "url": webpage.url,
                "created_at": webpage.created_at,
                "raw": webpage.raw,
            },
        )

    def create_insert_webpage_document(self, webpage: WebPage) -> list[str]:
        """
        Insert webpage document into Elasticsearch
        """
        document = self.create_webpage_document(webpage)
        return self.store.add_documents(documents=[document])

    def create_insert_webpage_documents(self, webpages: list[WebPage]) -> None:
        """
        Insert multiple webpage documents into Elasticsearch
        """
        documents = [self.create_webpage_document(webpage) for webpage in webpages]
        return self.store.add_documents(documents=documents)

    def create_insert_image_document(self, image: SourcedImageDescription) -> list[str]:
        """
        Insert image document into Elasticsearch
        """
        document = self.create_image_document(image)
        return self.store.add_documents(documents=[document])

    def create_insert_image_documents(
        self, images: list[SourcedImageDescription]
    ) -> None:
        """
        Insert multiple image documents into Elasticsearch
        """
        documents = [self.create_image_document(image) for image in images]
        return self.store.add_documents(documents=documents)

    def delete_document(self, document_id: str) -> None:
        """
        Delete document from Elasticsearch
        """
        return self.store.delete(ids=[document_id])

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete multiple documents from Elasticsearch
        """
        return self.store.delete(ids=document_ids)

    def similarity_search(self, query: str, **kwargs) -> list[Document]:
        """
        Search Elasticsearch for similar documents
        """
        return self.store.similarity_search(query=query, **kwargs)

    def find_document_by_url(self, url: str) -> dict:
        """
        Find document by URL
        """
        # elasticsearch query looking at metadata field url for exact match
        return self.elasticsearch.search(
            index=self.index_name,
            body={"query": {"term": {"metadata.url.keyword": {"value": url}}}},
        )
