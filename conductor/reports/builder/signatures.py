"""
Signatures for report building
"""
import dspy
from conductor.reports.builder import models
from typing import Union


class ConversationTurn(dspy.Signature):
    """
    You are an expert in the topic of the research purposed by the researcher.
    Use the conversation topic to ground the conversation.
    Your goal is to drive more compelling research and branch out to new potential topics.
    Use the conversation history to guide the conversation.
    This is a single interaction in the conversation.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    topic: str = dspy.InputField(prefix="Topic: ")
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    input_support: Union[models.CitedAnswerWithCredibility, None] = dspy.InputField(
        prefix="Input Support: "
    )
    response: str = dspy.OutputField(prefix="Expert's Response: ")


class ResearcherResponse(dspy.Signature):
    """
    You are the researcher who is asking the research questions.
    You are responding to the expert's response.
    Use the conversation history and supporting documents to guide the response.
    Use the conversation topic to ground the response.
    Drive the conversation forward to find new interesting topics and close research gaps.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    topic = dspy.InputField(prefix="Topic: ")
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    response: str = dspy.InputField(prefix="Expert's Response: ")
    input_support: Union[models.CitedAnswerWithCredibility, None] = dspy.InputField(
        prefix="Input support: "
    )
    new_input: str = dspy.OutputField(prefix="Researcher's Updated Question: ")


class RefinedQuestion(dspy.Signature):
    """
    You are refining the research questions to be more specific and drive the research forward.
    Use the conversation history to guide the refinement.
    The refined question should both branch out to new compelling topics and close existing research gaps discovered during the conversation.
    """

    conversation_history: list[models.Interaction] = dspy.InputField(
        prefix="Conversation History: "
    )
    topic: str = dspy.InputField(prefix="Topic: ")
    input: str = dspy.InputField(prefix="Researcher's Input: ")
    refined_question: str = dspy.OutputField(prefix="Refined Question: ")


class SectionOutline(dspy.Signature):
    """
    You are generating an outline for a section of the report.
    Use the specification to ground the outline.
    Use the section title to guide the outline.
    the section content should be multi level headers and bullet points.
    the section headers should be marked by # and sub headers by ## and so on.
    content should be marked by - and sub content by -- and so on.
    """

    specification: str = dspy.InputField(prefix="Specification: ")
    section_title: str = dspy.InputField(prefix="Section Title: ")
    section_outline: models.SectionOutline = dspy.OutputField()
