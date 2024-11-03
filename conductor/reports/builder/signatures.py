"""
Signatures for report building
"""
import dspy
from conductor.reports.builder import models


class ConversationTurn(dspy.Signature):
    """
    You are an expert in the topic of the research questions.
    You want to refine the research questions to be more specific through a conversation with the researcher.
    Use the conversation history to guide the conversation.
    This is a single interaction in the conversation.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    supporting_documents: list[models.CitedAnswerWithCredibility] = dspy.InputField(
        prefix="Supporting Documents: "
    )
    response: str = dspy.OutputField(prefix="Expert's Response: ")


class ResearcherResponse(dspy.Signature):
    """
    You are the researcher who is asking the research questions.
    You are responding to the expert's response.
    Use the conversation history and supporter documents to guide the response.
    Refine your responses to be more specific and drive the research forward that will be used to query the internet for more information.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    response: str = dspy.InputField(prefix="Expert's Response: ")
    supporting_documents: list[models.CitedAnswerWithCredibility] = dspy.InputField(
        prefix="Supporting Documents: "
    )
    new_input: str = dspy.OutputField(prefix="Researcher's Updated Question: ")


class RefinedQuestion(dspy.Signature):
    """
    You are refining the research questions to be more specific and drive the research forward.
    Use the conversation history to guide the refinement.
    The final refined question will be used to query the internet for more information that builds on the conversation and branches out to new topics.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    refined_question: str = dspy.OutputField(prefix="Refined Question: ")
