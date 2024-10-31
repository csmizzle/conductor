"""
DSPy retriever module for Evrim for custom RAG pipeline
"""
from conductor.rag.client import ElasticsearchRetrieverClient
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import dspy
from typing import List, Optional
import cohere


class ElasticRMClient(dspy.Retrieve):
    def __init__(
        self,
        elasticsearch: Elasticsearch,
        embeddings: Embeddings,
        index_name: str,
        cohere_api_key: str = None,
        k: int = 3,
    ) -> None:
        super().__init__(k=k)
        self.client = ElasticsearchRetrieverClient(
            elasticsearch=elasticsearch,
            embeddings=embeddings,
            index_name=index_name,
        )
        self.cohere_api_key = cohere_api_key

    def _rerank(
        self,
        query: str,
        documents: List[Document],
        model: str = "rerank-english-v3.0",
        top_n: int = 3,
    ) -> dict:
        """Rerank documents with Cohere API

        Args:
            query (str): initial search query
            documents (List[Document]): documents to rerank
            model (_type_, optional): cohere model to use. Defaults to "rerank-english-v3.0"top_n:int=3.

        Returns:
            dict: _description_
        """
        reranked_documents = []
        co = cohere.client_v2.ClientV2(self.cohere_api_key)
        content = [document.page_content for document in documents]
        reranked_indexes = co.rerank(
            query=query,
            model=model,
            documents=content,
            top_n=top_n,
        )
        for result in reranked_indexes.results:
            reranked_documents.append(documents[result.index])
        return reranked_documents

    def _format_documents(self, documents: List[Document]) -> List[dict]:
        transformed_documents = []
        for document in documents:
            transformed_documents.append(
                f"Source: {document.metadata["url"]}\nSource Content: {document.page_content}\n"
            )
        return transformed_documents

    def _rerank_documents(self, query: str, **kwargs) -> List[Document]:
        # get initial documents
        initial_documents = self.client.similarity_search(query=query, k=10)
        # use cohere to rerank documents
        reranked_documents = self._rerank(query=query, documents=initial_documents)
        return reranked_documents

    def forward(self, query: str, k: Optional[int] = 3, **kwargs) -> dspy.Prediction:
        if self.cohere_api_key:
            documents = self._rerank_documents(query, k=k)
        else:
            documents = self.client.similarity_search(query=query, k=k)
        transformed_documents = self._format_documents(documents)
        return dspy.Prediction(documents=transformed_documents)
