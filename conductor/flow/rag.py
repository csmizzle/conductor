"""
Answer generation using DSPy
- Each answer should have citations, faithfulness, and factual correctness
"""
import dspy
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.signatures import CitedAnswer
from conductor.flow.credibility import SourceCredibility, get_source_credibility
from pydantic import BaseModel, Field


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
    ) -> CitedAnswer:
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
