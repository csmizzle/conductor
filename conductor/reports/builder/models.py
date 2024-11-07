from pydantic import BaseModel, Field
from conductor.builder.agent import ResearchAgentTemplate
from conductor.flow.rag import CitedAnswerWithCredibility
from typing import Union


class Interaction(BaseModel):
    input: str = Field(description="The researcher's input")
    input_support: Union[CitedAnswerWithCredibility, None] = Field(
        description="The Input support"
    )
    response: str = Field(description="The writer's response")


class Conversation(BaseModel):
    topic: str = Field(description="The conversation topic")
    conversation_history: list[Interaction] = Field(
        description="The conversation history"
    )
    question: str = Field(description="The researcher's updated question")


class ResearchAgentConversations(BaseModel):
    agent: ResearchAgentTemplate = Field(description="The research agent")
    conversations: list[Conversation] = Field(
        description="The conversations with refined questions"
    )


class SectionOutline(BaseModel):
    section_title: str = Field(description="The section title")
    section_content: str = Field(description="The section content")


class ReportOutline(BaseModel):
    report_title: str = Field(description="The report title")
    report_sections: list[SectionOutline] = Field(description="The report sections")


class ConversationReport(BaseModel):
    outline: ReportOutline = Field(description="The report outline")
    conversations: list[ResearchAgentConversations] = Field(
        description="The research agent conversations"
    )


class Sentence(BaseModel):
    content: str = Field(description="The sentence content from the question")
    question: str = Field(description="The question for the sentence")


class SentenceWithAnswer(Sentence):
    answer: CitedAnswerWithCredibility = Field(description="The sentence answer")


class Paragraph(BaseModel):
    sentences: list[Sentence] = Field(description="The paragraph sentences")


class SourcedParagraph(BaseModel):
    sentences: list[SentenceWithAnswer] = Field(description="The paragraph sentences")


class Section(BaseModel):
    title: str = Field(description="The section title")
    paragraphs: list[Paragraph] = Field(description="The section paragraphs")


class SourcedSection(BaseModel):
    title: str = Field(description="The section title")
    paragraphs: list[SourcedParagraph] = Field(description="The section paragraphs")


class Report(BaseModel):
    sections: list[SourcedSection] = Field(description="The report sections")
