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
Use to following customer inputs to create an egagement strategy for this product:

Product: Conductor - an automated market research tool

Be sure to include best information of each company and person in the response.
  - Also include any specific links that will help with engagement.

Include an entry for each person in the Person Search Results section.

Break the response into the following sections:
  - Key Players
    - Add any additional key players that may be relevant
  - Company Backgrounds
    - Actionable background information relevant to the engagement strategy, include at least 2-3 sentences for each company.
  - Engagement Strategy
    - Break down the engagement strategy for each person for pitching them how Conductor can help them.
    - Be specific about the strategy for each person, include at least 2-3 sentences for each person.
  - Key URLs
    - Company Website
    - LinkedIn
    - Twitter

Person Search Results:
{apollo_people_data}
\n
{format_instructions}
"""
