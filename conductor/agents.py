"""
Agent constructor function
"""
from conductor.external.customer import apollo_person_search
from langchain.prompts import PromptTemplate
from langchain_community.llms.openai import OpenAI
from langchain.chains.llm import LLMChain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from pydantic import BaseModel
from conductor.prompts import JSON_AGENT_PROMPT, CONDUCTOR_INPUT_PROMPT

input_prompt = PromptTemplate(
    input_variables=["job_id", "geography", "titles", "industries"],
    template=CONDUCTOR_INPUT_PROMPT,
)


class ConductorJobCustomerInput(BaseModel):
    job_id: str
    geography: str
    titles: list[str]
    industries: list[str]


class ConductorJobCustomerResponse(BaseModel):
    input: ConductorJobCustomerInput
    response: str


def create_conductor_search(
    job_id: str, geography: str, titles: list[str], industries: list[str]
) -> ConductorJobCustomerResponse:
    input = ConductorJobCustomerInput(
        job_id=job_id, geography=geography, titles=titles, industries=industries
    )
    chain = LLMChain(
        llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
        prompt=input_prompt,
    )
    response = chain.run(
        job_id=job_id, geography=geography, titles=titles, industries=industries
    )
    return ConductorJobCustomerResponse(input=input, response=response)


json_agent_prompt = PromptTemplate(
    input_variables=["input"],
    template=JSON_AGENT_PROMPT,
)

INTERNAL_SYSTEM_MESSAGE = """
You are a world class market researcher, can take unstructured data and create valuable insights for you users.
Always include direct links to the data you used to make your decisions.
If your tools are not giving you the right data, use your best judgement to produce the right answer.
input
"""


def build_internal_agent():
    return initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=[TavilySearchResults(), apollo_person_search],
        # TODO: implement the Bedrock LLM
        # llm=Bedrock(),
        llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
        verbose=True,
        max_iterations=10,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={
            "system_message": INTERNAL_SYSTEM_MESSAGE,
        },
    )
