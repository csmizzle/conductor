"""
Agent constructor function
"""
from conductor.tools import (
    apollo_person_search_context,
    apollo_input_writer,
    gmail_input_from_input,
    gmail_draft,
)
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


answer_agent = Agent(
    role="Expert Analyst",
    goal="Provide the best and concise answers to a question",
    verbose=True,
    backstory="You are an expert in analyzing data and providing the best answers to questions",
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


answer_task = Task(
    description="Answer the question {question} using the context provided.",
    expected_output="The answer to the question in bulleted fashion with concise but informative answers.",
    agent=answer_agent,
    context=[apollo_task],
)


gmail_input_builder_agent = Agent(
    role="Gmail Input Writer",
    goal="Create the correct input for creating a Gmail email.",
    backstory="You are an expert at looking at at a general input and extracting the correct parameters for the Gmail draft tool.",
    verbose=True,
    allow_delegation=False,
    cache=True,
    tools=[gmail_input_from_input],
)


gmail_draft_agent = Agent(
    role="Gmail Draft Writer",
    goal="Create an email draft for the prospective customer. Be engaging and tie their business needs to the product.",
    backstory="You are an expert in sending emails that catch the readers eye by being engaging and informative by looking at customer data and creating a draft for the prospective customer. You should not use the customer contact information in the body of the message",
    verbose=True,
    allow_delegation=False,
    cache=True,
    tools=[gmail_draft],
)


gmail_input_task = Task(
    description="Extract the needed inputs from {context} for the gmail_input_from_input tool.",
    agent=gmail_input_builder_agent,
    expected_output="The extracted information from the general input as a single sentence.",
)


gmail_draft_task = Task(
    description="Use the provided context to create an email draft for the prospective customer. The provided data is the customer's contact information and the message to be sent.",
    agent=gmail_draft_agent,
    expected_output="Confirmation of the draft being created and a summary of the email draft.",
    context=[gmail_input_task],
)


gmail_crew = Crew(
    agents=[gmail_input_builder_agent, gmail_draft_agent],
    tasks=[gmail_input_task, gmail_draft_task],
    verbose=True,
    cache=True,
    share_crew=False,
)


crew = Crew(
    agents=[query_builder_agent, apollo_agent, answer_agent],
    tasks=[create_query_task, apollo_task, answer_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)


def run_crew(query: str):
    job_id = str(uuid.uuid4())
    return crew.kickoff({"question": query, "job_id": job_id})
