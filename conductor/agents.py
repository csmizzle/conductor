"""
Agent constructor function
"""
from conductor.external.customer import apollo_person_search
from conductor.prompts import INTERNAL_SYSTEM_MESSAGE
from conductor.llms import claude_v2_1
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory


def build_internal_agent():
    return initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=[apollo_person_search],
        llm=claude_v2_1,
        verbose=True,
        max_iterations=10,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={
            "system_message": INTERNAL_SYSTEM_MESSAGE,
        },
    )
