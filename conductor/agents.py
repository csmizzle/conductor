"""
Agent constructor function
"""
from conductor.tools import (
    apollo_pinecone_gpt4,
    discord_pinecone_gpt4,
    apollo_person_search,
    apollo_input_writer,
)
from conductor.prompts import INTERNAL_SYSTEM_MESSAGE
from conductor.llms import claude_v2_1
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from crewai import Agent, Task, Crew
import uuid


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


query_builder_agent = Agent(
    role="Translate a natural language query into a structured query for apollo",
    goal="Create a one sentence query for Apollo's person search tool",
    verbose=True,
    backstory="You are an expert in translating natural language queries into structured queries for Apollo's person search tool",
    tools=[apollo_input_writer],
)

retriever_agent = Agent(
    role="Vector Database Retriever",
    goal="Retrieve information from available vector databases",
    verbose=True,
    memory=True,
    backstory=(
        "You are capable of looking at natural language instructions and querying vector data bases for the best answer"
    ),
    tools=[apollo_pinecone_gpt4, discord_pinecone_gpt4],
    allow_delegation=False,
    cache=True,
)


apollo_agent = Agent(
    role="Apollo Person Searcher",
    goal="Retrieve information from Apollo's person search tool",
    verbose=True,
    memory=True,
    backstory=(
        "You are capable of looking at natural language instructions and querying Apollo's person search tool for the best answer"
        "Once you finish the search, you will store the results in a database for future reference."
        "Upon successful completion, you will provide the job_id used to track the information."
    ),
    tools=[apollo_person_search],
    allow_delegation=False,
    cache=True,
)


create_query_task = Task(
    description="Take this {question} and {job_id} and create a structured query for Apollo's person search tool. Always check with a human that your query is correct.",
    expected_output="A one sentence query for Apollo's person search tool.",
    agent=query_builder_agent,
    human_input=True,
)


retrieve_task = Task(
    description=(
        "Use {question} to craft a query for internal and external data sources."
        "Focus on identifying the best course of action for each query. "
        "The response should be insightful and actionable."
    ),
    expected_output="A bulleted list of external and internal data merged together in a report highlighting the best course of action.This list should include all information on the potential customers, contact information, their backgrounds, and engagement strategies.",
    agent=retriever_agent,
)


query_task = Task(
    description=(
        "Use {question} to craft a query for internal and external data sources."
        "Focus on identifying the best course of action for each query."
        "The response should be insightful and actionable."
    ),
    expected_output="A bulleted list of external and internal data merged together in a report highlighting the best course of action. This list should include all information on the potential customers, contact information, their backgrounds, and engagement strategies.",
    agent=retriever_agent,
)


apollo_task = Task(
    description="Search for new customer data in Apollo's person search tool. Use {job_id} to store the results in a database for future reference. Once finished provide the job_id used to track the information. Make sure with a human that the apollo search is good before you send it.",
    expected_output="A status of the apollo job with the job_id.",
    agent=apollo_agent,
    context=[create_query_task],
)


crew = Crew(
    agents=[query_builder_agent, apollo_agent, retriever_agent],
    tasks=[create_query_task, apollo_task, query_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)


def run_task_crew(query: str):
    job_id = str(uuid.uuid4())
    return crew.kickoff({"question": query, "job_id": job_id})
