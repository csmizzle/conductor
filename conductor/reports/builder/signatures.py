"""
Signatures for report building
"""
import dspy
from conductor.reports.builder import models
from conductor.flow.rag import CitedAnswerWithCredibility


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
    # input_support: Union[models.CitedAnswerWithCredibility, None] = dspy.InputField(
    #     prefix="Input Support: "
    # )
    response: str = dspy.OutputField(prefix="Expert's Response: ")


class ConversationSummary(dspy.Signature):
    """
    Summarize the conversation between the researcher and the expert.
    Make sure the summary captures the essence of the conversation and the research gaps and new topics that were discovered.
    This summary will be used to refine a general outline for a research report.
    """

    conversation_to_summarize: models.SlimConversation = dspy.InputField(
        prefix="Conversation: "
    )
    summary: str = dspy.OutputField(prefix="Summary: ")


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
    # input_support: Union[models.CitedAnswerWithCredibility, None] = dspy.InputField(
    #     prefix="Input support: "
    # )
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


class RefindedOutline(dspy.Signature):
    """
    You are refining the outline based on the conversation summaries and the new topics, research gaps, and insights discovered.
    Use the draft outline to as the structure of the refined outline.
    Maintain the headers and sub headers from the draft outline.
    Creatively blend the perspective into of the outline content.
    Use the topics and content of the conversations to drive the refinement of the outline content.
    The refined outline should both branch out to new compelling topics and close existing research gaps discovered during the conversation.
    """

    # conversation_summaries: list[str] = dspy.InputField(
    #     prefix="Conversation Summaries: "
    # )
    perspective: str = dspy.InputField(prefix="Perspective: ")
    draft_outline: models.ReportOutline = dspy.InputField(prefix="Draft Outline: ")
    refined_outline: models.ReportOutline = dspy.OutputField(prefix="Refined Outline: ")


class SectionQuestion(dspy.Signature):
    """
    You are generating questions for a vector database that will get answers that will be transformed into sentences for a section of the report.
    Use the section outline title to guide the questions.
    Use the section outline content to tailor the questions to ensure that the answers will be relevant to the section.
    The answers of the questions will be transformed into sentences for the section.
    The question should lend themselves to being transformed into sentences that will create a coherent section with a narrative flow.
    """

    section_outline_title: str = dspy.InputField(prefix="Section Outline Title: ")
    section_outline_content: str = dspy.InputField(prefix="Section Outline Content: ")
    questions: list[str] = dspy.OutputField(prefix="Questions: ")


class Section(dspy.Signature):
    """
    You are generating a section of the report.
    The section should be constructed from the section outline title and have a similar structure to the section outline content.
    Paragraphs should be constructed from sentences and not filled with any fluffy content.
    Paragraphs should be at least 3 sentences long but no longer than 5 sentences.
    Paragraphs never have duplicate content.
    Each sentence of the report is a transformation of a cited answer into a sentence.
    The transformed sentence should be accompanied by its original question and answer.
    Add analytical insights, they should be included in the section.
    Sections should be logically structured and flow from one sentence to the next with a logical progression.
    Section never have duplicate content.
    Use source credibility to highlight areas where you have high confidence in the answer and low confidence in the answer.
    Use words like "unlikely", "likely", or "highly likely" to indicate the level of confidence in the answer.
    """

    section_outline_title: str = dspy.InputField(prefix="Section Outline Title: ")
    section_outline_content: str = dspy.InputField(prefix="Section Outline Content: ")
    questions: list[str] = dspy.InputField(prefix="Questions: ")
    answers: list[CitedAnswerWithCredibility] = dspy.InputField(prefix="Answers: ")
    section: models.Section = dspy.OutputField(prefix="Section: ")


class ReportForReview(dspy.Signature):
    """
    Edit the report to make sure that all sections are not fluffy and have a clear narrative flow.
    Ensure the perspective is woven into the report.
    There should be no duplicate content in the report so use the answers for the sentences to ensure this.
    Duplicate language should be removed and replaced with synonyms.
    If the section is lacking second level analysis that would generate the reader novel insights, add this analysis.
    If the section is lacking a clear narrative flow, add this narrative flow.
    The generated section list is always the same length as the original section list, never add or remove sections.
    """

    sections: list[list[str]] = dspy.InputField(
        prefix="Report: ", desc="The report sections"
    )
    perspective: str = dspy.InputField(prefix="Perspective: ")
    edited_report: list[list[str]] = dspy.OutputField(
        prefix="Edited Report: ", desc="The edited report sections"
    )


class SectionForReview(dspy.Signature):
    """
    Edit the report to make sure that all sections are not fluffy and have a clear narrative flow.
    Ensure the perspective is woven into the report.
    There should be no duplicate content in the report so use the answers for the sentences to ensure this.
    If the section is lacking second level analysis that would generate the reader novel insights, add this analysis.
    If the section is lacking a clear narrative flow, add this narrative flow.
    The generated section list is always the same length as the original section list.
    """

    section: list[str] = dspy.InputField(prefix="Section: ")
    perspective: str = dspy.InputField(prefix="Perspective: ")
    edited_section: list[str] = dspy.OutputField(prefix="Edited Section: ")
