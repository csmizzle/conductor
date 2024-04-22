"""
Agent constructor function
"""
from conductor.tools import apollo_person_search
from conductor.tools import apollo_pinecone_gpt4, discord_pinecone_gpt4
from conductor.chains import create_apollo_input
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


knowledge_master = Agent(
    role="Knowledge Master",
    goal="Retrieve information from all available data to get the best answer",
    verbose=True,
    memory=True,
    backstory=(
        "Merge all available data sources to provide the best answer to any question."
        "Create insightful and actionable responses based on the data available."
    ),
    tools=[apollo_pinecone_gpt4, discord_pinecone_gpt4],
    allow_delegation=False,
)

query_task = Task(
    description=(
        "Use {question} to craft a query for internal and external data sources."
        "Focus on identifying the best course of action for each query. "
        "The response should be insightful and actionable."
    ),
    expected_output="A bulleted list of external and internal data merged together in a report highlighting the best course of action.This list should include all information on the potential customers, contact information, their backgrounds, and engagement strategies.",
    tools=[apollo_pinecone_gpt4, discord_pinecone_gpt4],
    agent=knowledge_master,
)

question_crew = Crew(
    agents=[knowledge_master],
    tasks=[query_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
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
)


query_task = Task(
    description=(
        "Use {question} to craft a query for internal and external data sources."
        "Focus on identifying the best course of action for each query."
        "The response should be insightful and actionable."
    ),
    expected_output="A bulleted list of external and internal data merged together in a report highlighting the best course of action.This list should include all information on the potential customers, contact information, their backgrounds, and engagement strategies.",
    agent=retriever_agent,
)


apollo_task = Task(
    description="Use {apollo_input} to search for new customer data in Apollo's person search tool. Use {job_id} to store the results in a database for future reference. Once finished provide the job_id used to track the information.",
    expected_output="A status of the apollo job with the job_id.",
    agent=apollo_agent,
)

crew = Crew(
    agents=[apollo_agent, retriever_agent],
    tasks=[apollo_task, query_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)


def run_task_crew(query: str):
    job_id = str(uuid.uuid4())
    apollo_input = create_apollo_input(query=query, job_id=job_id)
    return crew.kickoff(
        {"apollo_input": apollo_input["text"], "question": query, "job_id": job_id}
    )
