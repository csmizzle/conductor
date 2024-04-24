"""
Agent constructor function
"""
from conductor.tools import apollo_person_search_context, apollo_input_writer
from crewai import Agent, Task, Crew
import uuid


query_builder_agent = Agent(
    role="Translate a natural language query into a structured query for apollo",
    goal="Create a one sentence query for Apollo's person search tool",
    verbose=True,
    backstory="You are an expert in translating natural language queries into structured queries for Apollo's person search tool",
    tools=[apollo_input_writer],
    allow_delegation=False,
    cache=True,
)


apollo_agent = Agent(
    role="Apollo Person Searcher",
    goal="Retrieve information from Apollo's person search tool",
    verbose=True,
    backstory=(
        "You are capable of looking at natural language instructions and querying Apollo's person search tool for the best answer"
        "Once you finish the search, you will store the results in a database for future reference."
        "Upon successful completion, you will provide the job_id used to track the information."
    ),
    tools=[apollo_person_search_context],
    allow_delegation=False,
    cache=True,
)


create_query_task = Task(
    description="Take this {question} and {job_id} and create a structured query for Apollo's person search tool. Always check with a human that your query is correct.",
    expected_output="A one sentence query for Apollo's person search tool.",
    agent=query_builder_agent,
)


apollo_task = Task(
    description="Search for new customer data in Apollo's person search tool. Use {job_id} to store the results in a database for future reference. Once finished provide the job_id used to track the information. Make sure with a human that the apollo search is good before you send it.",
    expected_output="A status of the apollo job with the job_id.",
    agent=apollo_agent,
    context=[create_query_task],
)


crew = Crew(
    agents=[query_builder_agent, apollo_agent],
    tasks=[create_query_task, apollo_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)


def run_crew(query: str):
    job_id = str(uuid.uuid4())
    return crew.kickoff({"question": query, "job_id": job_id})
