"""
Answer generation using DSPy
- Each answer should have citations, faithfulness, and factual correctness
"""
import dspy
from conductor.flow.retriever import ElasticRMClient, ElasticDocumentIdRMClient
from conductor.flow.signatures import (
    CitedAnswer,
    CitedValue,
    QuestionHyde,
    ExtractValue,
)
from conductor.flow.models import CitedAnswer as CitedAnswerModel
from conductor.flow.credibility import SourceCredibility, get_source_credibility
from conductor.flow.models import NotAvailable
from conductor.crews.rag_marketing.tools import parallel_ingest, ingest
from conductor.rag.ingest import parallel_ingest_with_ids, ingest_with_ids
from serpapi import GoogleSearch
from pydantic import BaseModel, Field, InstanceOf
from typing import Union, Tuple, Optional
from loguru import logger
from langchain_core.embeddings import Embeddings
from elasticsearch import Elasticsearch
import os
import concurrent.futures


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


class CitationRAG(dspy.Module):
    def __init__(
        self,
        elastic_retriever: ElasticRMClient,
    ) -> None:
        super().__init__()
        self.retriever = elastic_retriever
        self.generate_answer = dspy.ChainOfThought(CitedAnswer)

    def forward(
        self,
        question: str,
    ) -> CitedAnswerWithCredibility:
        retrieved_documents = self.retriever(query=question)
        answer = self.generate_answer(question=question, documents=retrieved_documents)
        source_confidences = [
            get_source_credibility(source=source) for source in answer.answer.citations
        ]
        answer_with_credibility = CitedAnswerWithCredibility(
            question=question,
            answer=answer.answer.answer,
            documents=retrieved_documents.documents,
            citations=answer.answer.citations,
            faithfulness=answer.answer.faithfulness,
            factual_correctness=answer.answer.factual_correctness,
            confidence=answer.answer.confidence,
            answer_reasoning=answer.reasoning,
            source_credibility=[
                source_confidence.credibility
                for source_confidence in source_confidences
            ],
            source_credibility_reasoning=[
                source_confidence.reasoning for source_confidence in source_confidences
            ],
        )
        return answer_with_credibility


class AgenticCitationRAG(dspy.Module):
    def __init__(
        self,
        elastic_retriever: ElasticRMClient,
        max_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.retriever = elastic_retriever
        self.max_iterations = max_iterations
        self.generate_answer = dspy.ChainOfThought(CitedAnswer)
        self.hyde_question = dspy.Predict(QuestionHyde)

    def _retrieve_answer_from_internet(
        self, question: str, results: int = 5
    ) -> Tuple[CitedAnswerModel, list[str]]:
        """
        Retrieve the answer from the internet and index document in ElasticSearch
        """
        # first check the answer box of serp for the answer
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
            logger.info(f"Answer found in the answer box for question: {question}")
            logger.info(
                f"Ingesting answer link {google_results_dict['answer_box']['link']}"
            )
            ingest(
                url=google_results_dict["answer_box"]["link"],
                client=self.retriever.client,
            )
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
            retrieved_documents = dspy.Prediction(documents=retrieved_documents)
        elif "organic_results" in google_results_dict:
            urls_to_ingest = []
            for idx in range(min(results, len(google_results_dict["organic_results"]))):
                logger.info(f"Ingesting additional information for query {question}")
                urls_to_ingest.append(
                    google_results_dict["organic_results"][idx]["link"]
                )
            parallel_ingest(urls=urls_to_ingest, client=self.retriever.client)
            # search again to get the answer with the indexed documents
            retrieved_documents = self.retriever(query=question)
            answer = self.generate_answer(
                question=question, documents=retrieved_documents
            )
        return answer, retrieved_documents

    def forward(
        self,
        question: str,
    ) -> CitedAnswerWithCredibility:
        retrieved_documents = self.retriever(query=question)
        answer: CitedAnswerModel = self.generate_answer(
            question=question, documents=retrieved_documents
        )
        # agentic loop to find the answer within max_iterations
        if answer.answer.answer == NotAvailable.NOT_AVAILABLE.value:
            logger.info(f"Answer not available for question: {question}")
            for _ in range(self.max_iterations):
                # on first iteration, try to retrieve the answer from the internet
                if _ == 0:
                    answer, retrieved_documents = self._retrieve_answer_from_internet(
                        question
                    )
                    if answer.answer.answer != NotAvailable.NOT_AVAILABLE.value:
                        break
                # in subsequent iterations, use HyDE to modify the question and look against vector database with larger vector
                else:
                    logger.info(
                        f"Looking for answer using HyDE for question: {question}"
                    )
                    retrieved_documents = self.retriever(query=question)
                    answer = self.generate_answer(
                        question=question, documents=retrieved_documents
                    )
                    if answer.answer.answer != NotAvailable.NOT_AVAILABLE.value:
                        break
                    else:
                        logger.info(
                            f"Answer not available for question: {question}, modifying query with HyDE ..."
                        )
                        question = self.hyde_question(question=question).document
        source_confidences = [
            get_source_credibility(source=source) for source in answer.answer.citations
        ]
        answer_with_credibility = CitedAnswerWithCredibility(
            question=question,
            answer=answer.answer.answer,
            documents=retrieved_documents.documents,
            citations=answer.answer.citations,
            faithfulness=answer.answer.faithfulness,
            factual_correctness=answer.answer.factual_correctness,
            confidence=answer.answer.confidence,
            answer_reasoning=getattr(
                answer, "reasoning", None
            ),  # since not alwsys chain of thought
            source_credibility=[
                source_confidence.credibility
                for source_confidence in source_confidences
            ],
            source_credibility_reasoning=[
                source_confidence.reasoning for source_confidence in source_confidences
            ],
        )
        return answer_with_credibility


class AgenticCitationValueRAG(AgenticCitationRAG):
    """
    Convert the AgenticCitationRAG to transform sentences to values
    """

    def __init__(self, elastic_retriever, max_iterations=3):
        super().__init__(
            elastic_retriever=elastic_retriever, max_iterations=max_iterations
        )
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


class CitationValueRAG(dspy.Module):
    def __init__(
        self,
        elastic_retriever: ElasticRMClient,
    ) -> None:
        super().__init__()
        self.retriever = elastic_retriever
        self.generate_value = dspy.ChainOfThought(CitedValue)

    def forward(
        self,
        question: str,
    ) -> CitedValueWithCredibility:
        retrieved_documents = self.retriever(query=question)
        value = self.generate_value(question=question, documents=retrieved_documents)
        source_confidences = [
            get_source_credibility(source=source) for source in value.value.citations
        ]
        value_with_credibility = CitedValueWithCredibility(
            question=question,
            value=value.value.value,
            documents=retrieved_documents.documents,
            citations=value.value.citations,
            faithfulness=value.value.faithfulness,
            factual_correctness=value.value.factual_correctness,
            confidence=value.value.confidence,
            value_reasoning=value.reasoning,
            source_credibility=[
                source_confidence.credibility
                for source_confidence in source_confidences
            ],
            source_credibility_reasoning=[
                source_confidence.reasoning for source_confidence in source_confidences
            ],
        )
        return value_with_credibility


class WebSearchRAG(dspy.Module):
    def __init__(
        self,
        elastic_id_retriever: ElasticDocumentIdRMClient,
    ) -> None:
        super().__init__()
        self.retriever = elastic_id_retriever
        self.generate_answer = dspy.ChainOfThought(CitedAnswer)

    def _retrieve_answer_from_internet(
        self, question: str, results: int = 5
    ) -> Tuple[CitedAnswerModel, list[str]]:
        """
        Retrieve the answer from the internet and index document in ElasticSearch
        """
        # first check the answer box of serp for the answer
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
            logger.info(f"Answer found in the answer box for question: {question}")
            logger.info(
                f"Ingesting answer link {google_results_dict['answer_box']['link']}"
            )
            ingest_with_ids(
                url=google_results_dict["answer_box"]["link"],
                client=self.retriever.client,
            )
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
            retrieved_documents = dspy.Prediction(documents=retrieved_documents)
        elif "organic_results" in google_results_dict:
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
        return answer, retrieved_documents

    def forward(self, question: str) -> CitedAnswerWithCredibility:
        answer, retrieved_documents = self._retrieve_answer_from_internet(question)
        source_confidences = [
            get_source_credibility(source=source) for source in answer.answer.citations
        ]
        answer_with_credibility = CitedAnswerWithCredibility(
            question=question,
            answer=answer.answer.answer,
            documents=retrieved_documents.documents,
            citations=answer.answer.citations,
            faithfulness=answer.answer.faithfulness,
            factual_correctness=answer.answer.factual_correctness,
            confidence=answer.answer.confidence,
            answer_reasoning=getattr(
                answer, "reasoning", None
            ),  # since not always chain of thought
            source_credibility=[
                source_confidence.credibility
                for source_confidence in source_confidences
            ],
            source_credibility_reasoning=[
                source_confidence.reasoning for source_confidence in source_confidences
            ],
        )
        return answer_with_credibility

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
