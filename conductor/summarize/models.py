from conductor.flow.rag import DocumentWithCredibility
from pydantic import BaseModel


class SummarizedDocument(BaseModel):
    document: DocumentWithCredibility
    summary: str


class SummarizedDocuments(BaseModel):
    """
    A collection of summarized documents
    """

    documents: list[SummarizedDocument]
    summary: str
