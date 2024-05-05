from langchain.prompts import PromptTemplate
from conductor.parsers import (
    engagement_strategy_parser,
    gmail_input_parser,
    html_summary_parser,
)

JSON_AGENT_PROMPT = """
You are a world class market researcher, can take unstructured data and create valuable insights for you users.
You always use open source data from Wikipedia along with tools like apollo-person-search.
Always input the correct job_id into the tools that need it.
Look at the json to best answer the input.

INPUT:
{input}

TOOLS:
{tools}

TOOL_NAMES:
{tool_names}


AGENT_SCRATCHPAD:
{agent_scratchpad}
"""


CONDUCTOR_INPUT_PROMPT = """
Use the following input to create an optimal natural language search query for a world class market researcher that is responsible
for finding customer leads in a specific geography.

This agent will use the apollo-person-search tool to find the right data to answer the question.
-------
Product: Conductor - an automated market research tool
JOB_ID: {job_id}
  - This field is used to storing the results in a database for future reference
GEOGRAPHY: {geography}
TITLES: {titles}
INDUSTRIES: {industries}
"""


CONDUCTOR_APOLLO_CUSTOMER_PROMPT = """
Use to following potential customer inputs JSON to create an engagement strategy for this product:

Product: Conductor - an automated market research tool

- Engagement Strategy
  - Break down the engagement strategy for each person for pitching them how Conductor can help them.
  - Be specific about the strategy for each person, include at least 2-3 sentences for each person.

- Reasoning
  - Include the reasoning behind the strategy for each person.
  - Provide reasoning to show your thought process in the response.

Person Search Results:
{apollo_people_data}
\n
{format_instructions}
"""


INTERNAL_SYSTEM_MESSAGE = """
Use the apollo-person-search tool to find the right data to answer the question.
Always include direct links to the data you used to make your decisions.
If your tools are not giving you the right data, use your best judgement to produce the right answer.
{input}
"""

APOLLO_INPUT_PROMPT = """
Extract Apollo input parameters from a general input string below and provide a one sentence natural language query for the Apollo person search tool.


Apollo Input Parameters:
- Titles
- Location

{general_input}
"""


APOLLO_WITH_JOB_ID_INPUT_PROMPT = """
Extract Apollo input parameters from a general input string below and provide a one sentence natural language query for the Apollo person search tool with job_id: {job_id}.


Apollo Input Parameters:
- Titles
- Location

{general_input}
"""


GMAIL_INPUT_PROMPT = """
Extract the following parameters a general input string below and provide the extracted information.
If subject isn't provided, create one that is relevant to the body of the email.
If to isn't provided, return "No recipient provided".

Parameters:
  - Email Address(es) if there are multiple
  - Subject
  - Body

{general_input}
\n
{format_instructions}
"""


SUMMARY_INPUT_PROMPT = """
Using the semi-cleaned text from a web page, create a summary of the content that distills what the content is about.

{raw}
\n
{format_instructions}
"""


EMAIL_PROMPT = """
Use the below context to write an email summary.
Start the body of the email with "Here is your research!".
The email body should be a world class summary containing all information with tone: {tone} from the context into paragraphs and sign off with the provided sign off {sign_off}.
The email should contain all links and contact information mentioned in the context in a bulleted list.
You are simply summarizing the context and providing a concise email body, not reaching out to anyone mentioned in the context.
--------
Context: {context}
--------
Example Format:

Here is your research!

<Name, Company>
<Summary>
<Links>
<Contact Information>

{sign_off}
"""

input_prompt = PromptTemplate(
    input_variables=["job_id", "geography", "titles", "industries"],
    template=CONDUCTOR_INPUT_PROMPT,
)


input_prompt = PromptTemplate(
    input_variables=["apollo_people_data"],
    template=CONDUCTOR_APOLLO_CUSTOMER_PROMPT,
    partial_variables={
        "format_instructions": engagement_strategy_parser.get_format_instructions()
    },
)


apollo_with_job_id_input_prompt = PromptTemplate(
    input_variables=["job_id", "general_input"],
    template=APOLLO_WITH_JOB_ID_INPUT_PROMPT,
)


apollo_input_prompt = PromptTemplate(
    input_variables=["general_input"],
    template=APOLLO_INPUT_PROMPT,
)


gmail_input_prompt = PromptTemplate(
    input_variables=["general_input"],
    template=GMAIL_INPUT_PROMPT,
    partial_variables={
        "format_instructions": gmail_input_parser.get_format_instructions()
    },
)


html_summary_prompt = PromptTemplate(
    input_variables=["raw"],
    template=SUMMARY_INPUT_PROMPT,
    partial_variables={
        "format_instructions": html_summary_parser.get_format_instructions()
    },
)


email_prompt = PromptTemplate(
    input_variables=["context", "sign_off", "tone"],
    template=EMAIL_PROMPT,
)
