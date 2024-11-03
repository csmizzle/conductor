from pydantic import BaseModel, Field
from conductor.builder.agent import ResearchAgentTemplate
from conductor.flow.rag import CitedAnswerWithCredibility
from typing import Union


class Interaction(BaseModel):
    input: str = Field(description="The researcher's input")
    supporting_documents: Union[list[CitedAnswerWithCredibility], list] = Field(
        description="The supporting documents"
    )
    response: str = Field(description="The writer's response")


class Conversation(BaseModel):
    conversation_history: list[Interaction] = Field(
        description="The conversation history"
    )
    question: str = Field(description="The researcher's updated question")


class ResearchAgentConversations(BaseModel):
    agent: ResearchAgentTemplate = Field(description="The research agent")
    conversations: list[Conversation] = Field(
        description="The conversations with refined questions"
    )
