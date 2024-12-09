from typing import Literal
from conductor.agents import signatures
from loguru import logger
import dspy


def grade_documents(state) -> Literal["generate", "rewrite"]:
    logger.info("Grading documents ...")
    # replace chains with signatures
    document_grader = dspy.ChainOfThought(signatures.DocumentGrader)
    # get question and content
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    doc = last_message.content
    # grade documents
    score = document_grader(question=question, content=doc)
    if score.score.relevance:
        logger.info("Document is relevant to the question, moving to generation ...")
        return "generate"
    else:
        logger.info("Document is not relevant to the question, rewriting ...")
        return "rewrite"
