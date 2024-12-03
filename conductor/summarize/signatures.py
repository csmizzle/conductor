from conductor.flow.rag import DocumentWithCredibility
import dspy


class DocumentSummary(dspy.Signature):
    """
    Look at a document and summarize it to the key parts of the document based on the question
    The summary should be a few sentences long.
    """

    question: str = dspy.InputField(description="The question to base the summary on")
    document: DocumentWithCredibility = dspy.InputField(
        description="The document to summarize"
    )
    summary: str = dspy.OutputField(description="The summary of the document")


class MasterSummary(dspy.Signature):
    """
    Summarize a collection of summaries to a single summary
    Use the question to base the summary on
    The final summary should be a few sentences to a few paragraphs long depending on the number of summaries.
    """

    question = dspy.InputField(description="The question to base the summary on")
    summaries: list[str] = dspy.InputField(description="The summaries of the documents")
    summary: str = dspy.OutputField(description="The final summary of the documents")
