"""
Agent constructor function
"""
from conductor.tools import (
    apollo_person_search_context_job,
    apollo_input_with_job_writer,
    apollo_input_writer,
    apollo_person_search_context,
    gmail_input_from_input,
    gmail_draft,
    gmail_send,
    apollo_email_sender,
)
from crewai import Agent, Task, Crew
import uuid


query_job_builder_agent = Agent(
    role="Translate a natural language query into a structured query for apollo",
    goal="Create a one sentence query for Apollo's person search tool",
    verbose=True,
    backstory="You are an expert in translating natural language queries into structured queries for Apollo's person search tool",
    tools=[apollo_input_with_job_writer],
    allow_delegation=False,
    cache=True,
)

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
    role="Apollo  Person Searcher",
    goal="Retrieve information from Apollo's person search tool",
    verbose=True,
    backstory=(
        "You are capable of looking at natural language instructions and querying Apollo's person search tool for the best answer"
        "Once you finish the search, you will store the results in a database for future reference."
        "Upon successful completion, you will provide the results of the search."
    ),
    tools=[apollo_person_search_context],
    allow_delegation=False,
    cache=True,
)


apollo_job_agent = Agent(
    role="Apollo Job Person Searcher",
    goal="Retrieve information from Apollo's person search tool",
    verbose=True,
    backstory=(
        "You are capable of looking at natural language instructions and querying Apollo's person search tool for the best answer"
        "Once you finish the search, you will store the results in a database for future reference."
        "Upon successful completion, you will provide the job_id used to track the information."
    ),
    tools=[apollo_person_search_context_job],
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


create_query_job_task = Task(
    description="Take this {question} and {job_id} and create a structured query for Apollo's person search tool. Always check with a human that your query is correct.",
    expected_output="A one sentence query for Apollo's person search tool.",
    agent=query_job_builder_agent,
)


create_query_task = Task(
    description="Take this {input} and and create a structured query for Apollo's person search tool. Always check with a human that your query is correct.",
    expected_output="A one sentence query for Apollo's person search tool.",
    agent=query_builder_agent,
)


apollo_job_task = Task(
    description="Search for new customer data in Apollo's person search tool. Use {job_id} to store the results in a database for future reference. Once finished provide the job_id used to track the information. Make sure with a human that the apollo search is good before you send it.",
    expected_output="A status of the apollo job with the job_id.",
    agent=apollo_job_agent,
    context=[create_query_job_task],
)

apollo_task = Task(
    description="Search for new customer data in Apollo's person search tool.",
    expected_output="The output of the search in a structured format.",
    agent=apollo_agent,
    context=[create_query_task],
)


answer_job_task = Task(
    description="Answer the question {question} using the context provided.",
    expected_output="The answer to the question in bulleted fashion with concise but informative answers.",
    agent=answer_agent,
    context=[apollo_job_task],
)


answer_task = Task(
    description="Answer the input {input} using the context provided.",
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


gmail_send_agent = Agent(
    role="Gmail Email Sender",
    goal="Create an email. The message should be structured, clean, and informative. Finish the email with Best, John Envoy.",
    backstory="You are an expert in sending emails using the provided input.",
    verbose=True,
    allow_delegation=False,
    cache=True,
    tools=[gmail_send],
)


gmail_input_task = Task(
    description="Extract the needed inputs from {context} for the gmail_input_from_input tool.",
    agent=gmail_input_builder_agent,
    expected_output="The extracted information from the general input as a single sentence.",
)


gmail_apollo_input_task = Task(
    description="Extract the needed inputs from {input} and the provided context for the gmail_input_from_input tool.",
    agent=gmail_input_builder_agent,
    expected_output="The extracted information from the general input as a single sentence.",
    context=[apollo_task],
)


gmail_draft_task = Task(
    description="Use the provided context to create an email draft for the prospective customer. The provided data is the customer's contact information and the message to be sent.",
    agent=gmail_draft_agent,
    expected_output="Confirmation of the draft being created and a summary of the email draft.",
    context=[gmail_input_task],
)

gmail_apollo_draft_task = Task(
    description="Use the provided context to create an email draft for the prospective customer. The provided data is the customer's contact information and the message to be sent.",
    agent=gmail_draft_agent,
    expected_output="Confirmation of the draft being created and a summary of the email draft.",
    context=[gmail_input_task],
)


gmail_apollo_send_task = Task(
    description="Use the provided context to send an email using the exact text. Sign off as Best, John Envoy.",
    agent=gmail_send_agent,
    expected_output="Confirmation of the email being sent and a summary of the email.",
    context=[gmail_apollo_input_task],
)

apollo_email_agent = Agent(
    role="Apollo Email Writer",
    goal="Create an email draft using the Apollo Person Search results and send to the recipient.",
    backstory="You are an expert in sending emails that catch the readers eye by being engaging and informative by looking at customer data and creating a draft for the prospective customer. You should not use the customer contact information in the body of the message",
    verbose=True,
    allow_delegation=False,
    cache=True,
    tools=[apollo_email_sender],
)

apollo_email_task = Task(
    description="Use {input} to create and send an email to the recipient.",
    agent=apollo_email_agent,
    expected_output="Confirmation of the email being sent and a summary of the email.",
)


answer_email_task = Task(
    description="Summarize the data in the context and given the input: {input}",
    expected_output="A summary of your actions with the data from the email in a bullet point format.",
    agent=answer_agent,
    context=[apollo_email_task],
)


gmail_crew = Crew(
    agents=[gmail_input_builder_agent, gmail_draft_agent],
    tasks=[gmail_input_task, gmail_draft_task],
    verbose=True,
    cache=True,
    share_crew=False,
)


crew = Crew(
    agents=[query_job_builder_agent, apollo_job_agent, answer_agent],
    tasks=[create_query_job_task, apollo_job_task, answer_job_task],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)

market_email_crew = Crew(
    agents=[
        query_builder_agent,
        apollo_email_agent,
        answer_agent,
    ],
    tasks=[
        create_query_task,
        apollo_email_task,
        answer_email_task,
    ],
    verbose=True,
    memory=True,
    cache=True,
    share_crew=False,
)


def run_crew(query: str):
    job_id = str(uuid.uuid4())
    return crew.kickoff({"question": query, "job_id": job_id})
