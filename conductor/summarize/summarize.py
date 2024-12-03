from conductor.flow.rag import DocumentWithCredibility
from conductor.summarize.signatures import (
    DocumentSummary,
    MasterSummary,
)
from conductor.summarize.models import (
    SummarizedDocument,
    SummarizedDocuments,
)
import dspy
import concurrent.futures
from loguru import logger


def summarize_document(
    document: DocumentWithCredibility,
    question: str,
) -> SummarizedDocument:
    logger.info(f"Summarizing document {document.source}")
    generate_summary = dspy.ChainOfThought(DocumentSummary)
    generated_summary = generate_summary(question=question, document=document).summary
    return SummarizedDocument(
        summary=generated_summary,
        document=document,
    )


def summarize_document_parallel(
    documents: list[DocumentWithCredibility],
    question: str,
) -> list[SummarizedDocument]:
    """
    Run summarize_document in parallel
    """
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(summarize_document, document, question)
            for document in documents
        ]
        for future in concurrent.futures.as_completed(futures):
            summaries.append(future.result())
    return summaries


def generate_master_summary(
    documents: list[DocumentWithCredibility],
    question: str,
) -> SummarizedDocuments:
    """
    Generate a master summary from the summaries of the documents
    """
    logger.info("Generating master summary")
    summarized_documents = summarize_document_parallel(documents, question)
    generate_master_summary = dspy.ChainOfThought(MasterSummary)
    # get the summaries from parallel processing
    summaries = [summary.summary for summary in summarized_documents]
    # create the master summary
    master_summary = generate_master_summary(
        question=question, summaries=summaries
    ).summary
    return SummarizedDocuments(
        summary=master_summary,
        documents=summarized_documents,
    )
