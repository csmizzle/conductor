"""
Ingestion logic for RAG data
- Ingest website data
- Clean data
- Vectorize data
- Store data
"""
from typing import List
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.rag.chunking import WebPageContentSplitter


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

    def create_webpage_document(self, webpage: WebPage) -> List[Document]:
        """
        Ingest webpage document into Elasticsearch
        """
        chunker = WebPageContentSplitter(webpage=webpage)
        return chunker.create_documents()

    def create_insert_webpage_document(self, webpage: WebPage) -> list[str]:
        """
        Insert webpage document into Elasticsearch
        """
        document = self.create_webpage_document(webpage)
        return self.store.add_documents(documents=document)

    def create_insert_webpage_documents(self, webpages: list[WebPage]) -> None:
        """
        Insert multiple webpage documents into Elasticsearch
        """
        documents = []
        created_documents = [
            self.create_webpage_document(webpage) for webpage in webpages
        ]
        for created_document in created_documents:
            documents.extend(created_document)
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

    def find_documents_by_url(
        self,
        url: str,
    ) -> list[dict]:
        """
        Find document by URL
        """
        # elasticsearch query looking at metadata field url for exact match
        body = {
            "query": {"term": {"metadata.url.keyword": url}},
        }
        results = self.elasticsearch.search(
            index=self.index_name,
            body=body,
            scroll="2m",  # Keep the scroll context alive for 2 minutes
            size=100,  # Batch size
        )
        all_hits = []
        scroll_id = results["_scroll_id"]
        hits = results["hits"]["hits"]
        # Continue scrolling until no more hits are returned
        while len(hits) > 0:
            for hit in hits:
                all_hits.append(
                    {**{"_id": hit["_id"]}, **hit["_source"]}
                )  # append _id to documents

            # Fetch the next batch of results
            response = self.elasticsearch.scroll(scroll_id=scroll_id, scroll="2m")
            hits = response["hits"]["hits"]
        # Clear the scroll context when done
        self.elasticsearch.clear_scroll(scroll_id=scroll_id)
        return all_hits

    def mget_documents(self, document_ids: list[str]) -> list[Document]:
        """
        Get multiple documents by ID
        """
        raw_documents = self.elasticsearch.mget(ids=document_ids, index=self.index_name)
        # create the langchain document objects
        return [
            Document(
                page_content=raw_document["_source"]["text"],
                metadata=raw_document["_source"]["metadata"],
            )
            for raw_document in raw_documents["docs"]
        ]

    def mget_raw(self, document_ids: list[str]) -> list[dict]:
        """
        Get multiple raw documents by ID
        """
        documents = self.mget_documents(ids=document_ids, index=self.index_name)
        return [
            "Source URL: "
            + document.metadata["url"]
            + "\n"
            + "Source Content: "
            + document.page_content
            for document in documents
        ]
