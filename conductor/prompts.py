from langchain.prompts import PromptTemplate


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

These leads should also be augmented with engagement strategies for each person found.

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
Once you have this data, use the person-engagement-strategy-tool to create an engagement strategy for each person.
Always include direct links to the data you used to make your decisions.
If your tools are not giving you the right data, use your best judgement to produce the right answer.
{input}
"""


input_prompt = PromptTemplate(
    input_variables=["job_id", "geography", "titles", "industries"],
    template=CONDUCTOR_INPUT_PROMPT,
)
