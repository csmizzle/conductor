"""
Answer generation using DSPy
- Each answer should have citations, faithfulness, and factual correctness
"""
import dspy
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.signatures import CitedAnswer, CitedValue
from conductor.flow.credibility import SourceCredibility, get_source_credibility
from conductor.flow.models import NotAvailable
from pydantic import BaseModel, Field
from typing import Union


class CitedAnswerWithCredibility(BaseModel):
    question: str = Field(description="The question")
    answer: str = Field(description="The answer for the question")
    documents: list[str] = Field(
        description="The documents used to generate the answer"
    )
    answer_reasoning: str = Field(description="The reasoning behind the answer")
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: float = Field(
        ge=0, le=1, description="The faithfulness of the answer"
    )
    factual_correctness: float = Field(
        ge=0, le=1, description="The factual correctness of the answer"
    )
    confidence: float = Field(ge=0, le=1, description="The confidence of the answer")
    source_credibility: list[SourceCredibility] = Field(
        description="The credibility of the sources"
    )
    source_credibility_reasoning: list[str] = Field(
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
    value_reasoning: str = Field(description="The reasoning behind the value")
    citations: list[str] = Field(description="The URLs used in the value")
    faithfulness: float = Field(ge=0, le=1, description="The faithfulness of the value")
    factual_correctness: float = Field(
        ge=0, le=1, description="The factual correctness of the value"
    )
    confidence: float = Field(ge=0, le=1, description="The confidence of the value")
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
