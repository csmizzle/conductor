import dspy
from conductor.agents import models
from conductor.flow.rag import DocumentWithCredibility


class DocumentGrader(dspy.Signature):
    """
    Grade a document based on the question and content
    """

    question: str = dspy.InputField(description="The question to grade against")
    content: str = dspy.InputField(description="The content to grade")
    score: models.Score = dspy.OutputField(
        description="Boolean field if the document is relevant to the question"
    )


class QuestionRewriter(dspy.Signature):
    """
    Rewrite a question to reason and improve the question based on sematic intent / meaning
    """

    question: str = dspy.InputField(description="The question to rewrite")
    rewritten_question: str = dspy.OutputField(description="The rewritten question")


class AnswerGenerator(dspy.Signature):
    """
    Generate an answer based on the question and documents
    """

    question: str = dspy.InputField(
        description="The question to generate an answer for"
    )
    document: DocumentWithCredibility = dspy.InputField(
        description="The documents to generate an answer from"
    )
    answer: str = dspy.OutputField(description="The generated answer")
