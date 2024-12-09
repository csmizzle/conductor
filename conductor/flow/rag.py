"""
Answer generation using DSPy
- Each answer should have citations, faithfulness, and factual correctness
"""
import dspy
from conductor.flow.retriever import ElasticDocumentIdRMClient
from conductor.flow.signatures import (
    CitedAnswer,
    ExtractValue,
    AnswerReasoning,
)
from conductor.flow.models import CitedAnswer as CitedAnswerModel
from conductor.flow.credibility import SourceCredibility, get_source_credibility
from conductor.flow.models import NotAvailable
from conductor.rag.ingest import (
    parallel_ingest_with_ids,
    ingest_with_ids,
)
from serpapi import GoogleSearch
from pydantic import BaseModel, Field, InstanceOf
from typing import Union, Tuple, Optional
from loguru import logger
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from elasticsearch import Elasticsearch
import os
import concurrent.futures


class SlimCitedAnswerWithCredibility(BaseModel):
    """
    Citation Answer with credibility without the documents
    """

    question: str = Field(description="The question")
    answer: str = Field(description="The answer for the question")
    answer_reasoning: Union[str, None] = Field(
        description="The reasoning behind the answer"
    )
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the answer")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the answer"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the answer")
    source_credibility: list[SourceCredibility] = Field(
        description="The credibility of the sources"
    )
    source_credibility_reasoning: Optional[list[str]] = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True


class CitedAnswerWithCredibility(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The answer for the question")
    documents: list[str] = Field(
        description="The documents used to generate the answer"
    )
    answer_reasoning: Union[str, None] = Field(
        description="The reasoning behind the answer"
    )
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the answer")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the answer"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the answer")
    source_credibility: list[SourceCredibility] = Field(
        description="The credibility of the sources"
    )
    source_credibility_reasoning: Optional[list[str]] = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True

    def slim(self) -> list[SlimCitedAnswerWithCredibility]:
        return SlimCitedAnswerWithCredibility(
            question=self.question,
            answer=self.answer,
            answer_reasoning=self.answer_reasoning,
            citations=self.citations,
            faithfulness=self.faithfulness,
            factual_correctness=self.factual_correctness,
            confidence=self.confidence,
            source_credibility=self.source_credibility,
            source_credibility_reasoning=self.source_credibility_reasoning,
        )


class CitedValueWithCredibility(BaseModel):
    """Best value for a question"""

    question: str = Field(description="The question")
    value: Union[str, bool, NotAvailable] = Field(
        description="The value for the question"
    )
    documents: list[str] = Field(description="The documents used to generate the value")
    value_reasoning: Union[str, None] = Field(
        description="The reasoning behind the value"
    )
    citations: list[str] = Field(description="The URLs used in the value")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the value")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the value"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the value")
    source_credibility: list[SourceCredibility] = Field(
        description="The credibility of the sources"
    )
    source_credibility_reasoning: list[str] = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True


class CitedBooleanWithCredibility(CitedValueWithCredibility):
    value: Union[bool, NotAvailable] = Field(description="The value for the question")


class DocumentWithCredibility(BaseModel):
    content: str = Field(description="The content of the document")
    source: str = Field(description="The source of the document")
    source_credibility: SourceCredibility = Field(
        description="The credibility of the source"
    )
    source_credibility_reasoning: str = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True


class DocumentIDRetriever(dspy.Module):
    @classmethod
    def with_elasticsearch_id_retriever(
        cls,
        embeddings: InstanceOf[Embeddings],
        elasticsearch: InstanceOf[Elasticsearch],
        index_name: str,
        cohere_api_key: str = None,
        k: int = 10,
        rerank_top_n: int = 5,
    ) -> "WebSearchRAG":
        retriever = ElasticDocumentIdRMClient(
            elasticsearch=elasticsearch,
            index_name=index_name,
            embeddings=embeddings,
            cohere_api_key=cohere_api_key,
            k=k,
            rerank_top_n=rerank_top_n,
        )
        return cls(elastic_id_retriever=retriever)


class WebDocumentRetriever(DocumentIDRetriever):
    def __init__(
        self,
        elastic_id_retriever: ElasticDocumentIdRMClient,
    ) -> None:
        super().__init__()
        self.retriever = elastic_id_retriever
        self.generate_answer = dspy.ChainOfThought(CitedAnswer)
        self.generate_answer_reasoning = dspy.ChainOfThought(AnswerReasoning)

    def _retrieve_documents_from_internet(
        self, question: str, results: int = 5
    ) -> list[Document]:
        """
        Retrieve the answer from the internet and index document in ElasticSearch
        """
        # first check the answer box of serp for the answer
        logger.info(f"Searching the internet for the answer to question: {question}")
        retrieved_documents = None
        search = GoogleSearch(
            {
                "q": question,
                "hl": "en",
                "gl": "us",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
        google_results_dict = search.get_dict()
        if "answer_box" in google_results_dict:
            if "link" in google_results_dict["answer_box"]:
                logger.info(
                    f"Ingesting answer link {google_results_dict['answer_box']['link']}"
                )
                documents = ingest_with_ids(
                    url=google_results_dict["answer_box"]["link"],
                    client=self.retriever.client,
                )
                if "snippet" in google_results_dict["answer_box"]:
                    logger.info("Answer found in the snippet")
                    # get source documents
                    if documents:
                        document_ids = []
                        for urls in documents.values():
                            document_ids.extend(urls)
                        retrieved_documents = self.retriever.get_documents(
                            query=question, document_ids=document_ids
                        )
                    else:
                        logger.info("No documents were ingested ...")
                else:
                    logger.info(
                        "No answer found in the snippet, ingesting more data ..."
                    )
            else:
                logger.info(
                    "No answer link found in the answer box, ingesting more data ..."
                )
        if not retrieved_documents and "organic_results" in google_results_dict:
            urls_to_ingest = []
            logger.info(f"Ingesting additional information for query {question}")
            for idx in range(min(results, len(google_results_dict["organic_results"]))):
                urls_to_ingest.append(
                    google_results_dict["organic_results"][idx]["link"]
                )
            documents = parallel_ingest_with_ids(
                urls=urls_to_ingest, client=self.retriever.client
            )
            # get the document ids
            if documents:
                document_ids = []
                for urls in documents.values():
                    document_ids.extend(urls)
                # search again to get the answer with the indexed documents
                retrieved_documents = self.retriever.get_documents(
                    query=question, document_ids=document_ids
                )
            else:
                logger.info("No documents were ingested ...")
        return retrieved_documents

    def forward(self, question: str) -> list[DocumentWithCredibility]:
        documents = []
        retrieved_documents = self._retrieve_documents_from_internet(question)
        if retrieved_documents:
            for document in retrieved_documents:
                # assign source credibility
                source_confidence = get_source_credibility(
                    source=document.metadata["url"]
                )
                document = DocumentWithCredibility(
                    content=document.page_content,
                    source=document.metadata["url"],
                    source_credibility=source_confidence.credibility,
                    source_credibility_reasoning=source_confidence.reasoning,
                )
                documents.append(document)
        else:
            logger.warning(f"No documents found for question: {question}")
        return documents


class WebSearchRAG(DocumentIDRetriever):
    def __init__(
        self,
        elastic_id_retriever: ElasticDocumentIdRMClient,
    ) -> None:
        super().__init__()
        self.retriever = elastic_id_retriever
        self.generate_answer = dspy.ChainOfThought(CitedAnswer)
        self.generate_answer_reasoning = dspy.ChainOfThought(AnswerReasoning)

    def _retrieve_answer_from_internet(
        self, question: str, results: int = 5
    ) -> Tuple[dspy.Prediction, list[str]]:
        """
        Retrieve the answer from the internet and index document in ElasticSearch
        """
        # first check the answer box of serp for the answer
        answer = None
        logger.info(f"Searching the internet for the answer to question: {question}")
        retrieved_documents = []
        search = GoogleSearch(
            {
                "q": question,
                "hl": "en",
                "gl": "us",
                "api_key": os.getenv("SERPAPI_API_KEY"),
            }
        )
        google_results_dict = search.get_dict()
        if "answer_box" in google_results_dict:
            logger.info(
                f"Ingesting answer link {google_results_dict['answer_box']['link']}"
            )
            documents = ingest_with_ids(
                url=google_results_dict["answer_box"]["link"],
                client=self.retriever.client,
            )
            if "snippet" in google_results_dict["answer_box"]:
                logger.info("Answer found in the snippet")
                answer = google_results_dict["answer_box"]["snippet"]
                url = google_results_dict["answer_box"]["link"]
                answer = dspy.Prediction(
                    answer=CitedAnswerModel(
                        answer=answer,
                        citations=[url],
                        faithfulness=5,
                        factual_correctness=5,
                        confidence=5,
                    )
                )
                # get source documents
                if documents:
                    document_ids = []
                    for urls in documents.values():
                        document_ids.extend(urls)
                    retrieved_documents = self.retriever(
                        query=question, document_ids=document_ids
                    )
                else:
                    logger.info("No documents were ingested ...")
            else:
                logger.info("No answer found in the snippet, ingesting more data ...")
        if not answer and "organic_results" in google_results_dict:
            urls_to_ingest = []
            logger.info(f"Ingesting additional information for query {question}")
            for idx in range(min(results, len(google_results_dict["organic_results"]))):
                urls_to_ingest.append(
                    google_results_dict["organic_results"][idx]["link"]
                )
            documents = parallel_ingest_with_ids(
                urls=urls_to_ingest, client=self.retriever.client
            )
            if documents:
                # get the document ids
                document_ids = []
                for urls in documents.values():
                    document_ids.extend(urls)
                # search again to get the answer with the indexed documents
                retrieved_documents = self.retriever(
                    query=question, document_ids=document_ids
                )
                answer = self.generate_answer(
                    question=question, documents=retrieved_documents
                )
            else:
                logger.info("No documents were ingested ...")
        return answer, retrieved_documents

    def _get_answer_reasoning(
        self,
        answer: CitedAnswerModel,
        question: str,
        retrieved_documents: dspy.Prediction,
    ) -> None:
        # generate answer reasoning if not available
        if not hasattr(answer, "reasoning"):
            logger.info(f"Generating reasoning for answer: {answer.answer.answer}")
            reasoning = self.generate_answer_reasoning(
                answer=answer.answer.answer,
                citations=answer.answer.citations,
                question=question,
                documents=retrieved_documents.documents
                if hasattr(retrieved_documents, "documents")
                else [],
                faithfulness=answer.answer.faithfulness,
                factual_correctness=answer.answer.factual_correctness,
                confidence=answer.answer.confidence,
            )
            return reasoning.reasoning
        else:
            return answer.reasoning

    def forward(self, question: str) -> CitedAnswerWithCredibility:
        answer, retrieved_documents = self._retrieve_answer_from_internet(question)
        # assign source credibility
        source_confidences = [
            get_source_credibility(source=source) for source in answer.answer.citations
        ]
        answer_with_credibility = CitedAnswerWithCredibility(
            question=question,
            answer=answer.answer.answer,
            documents=retrieved_documents.documents
            if hasattr(retrieved_documents, "documents")
            else [],
            citations=answer.answer.citations,
            faithfulness=answer.answer.faithfulness,
            factual_correctness=answer.answer.factual_correctness,
            confidence=answer.answer.confidence,
            # get answer reasoning if not there
            answer_reasoning=self._get_answer_reasoning(
                answer, question, retrieved_documents
            ),
            source_credibility=[
                source_confidence.credibility
                for source_confidence in source_confidences
            ],
            source_credibility_reasoning=[
                source_confidence.reasoning for source_confidence in source_confidences
            ],
        )
        return answer_with_credibility


class WebSearchValueRAG(WebSearchRAG):
    """
    Convert the AgenticCitationRAG to transform sentences to values
    """

    def __init__(self, elastic_id_retriever):
        super().__init__(elastic_id_retriever=elastic_id_retriever)
        self.generate_value = dspy.Predict(ExtractValue)

    def forward(self, question: str) -> CitedValueWithCredibility:
        # run the forward method of the parent class
        cited_answer = super().forward(question=question)
        # convert the answer to a value
        value = self.generate_value(question=question, answer=cited_answer.answer)
        return CitedValueWithCredibility(
            question=question,
            value=value.value,
            documents=cited_answer.documents,
            citations=cited_answer.citations,
            faithfulness=cited_answer.faithfulness,
            factual_correctness=cited_answer.factual_correctness,
            confidence=cited_answer.confidence,
            value_reasoning=cited_answer.answer_reasoning,
            source_credibility=cited_answer.source_credibility,
            source_credibility_reasoning=cited_answer.source_credibility_reasoning,
        )


def get_answer(
    question: str, rag: InstanceOf[dspy.Module]
) -> CitedAnswerWithCredibility:
    return rag(question=question)


def get_answers(
    questions: list[str], rag: InstanceOf[dspy.Module]
) -> list[CitedAnswerWithCredibility]:
    """
    Run get answers in parallel
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for question in questions:
            futures.append(executor.submit(get_answer, question, rag))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def get_value(question: str, rag: InstanceOf[dspy.Module]) -> CitedValueWithCredibility:
    return rag(question=question)


def get_values(
    questions: list[str], rag: InstanceOf[dspy.Module]
) -> list[CitedValueWithCredibility]:
    """
    Run get values in parallel
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for question in questions:
            futures.append(executor.submit(get_value, question, rag))
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results
