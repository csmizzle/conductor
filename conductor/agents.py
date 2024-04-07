"""
Agent constructor function
"""
from conductor.external.customer import apollo_person_search
from conductor.llms import claude_v2_1
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
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
        llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
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
Use the apollo-person-search tool to find the right data to answer the question.
Always include direct links to the data you used to make your decisions.
If your tools are not giving you the right data, use your best judgement to produce the right answer.
{input}
"""


def build_internal_agent():
    return initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=[apollo_person_search],
        # TODO: implement the Bedrock LLM
        # llm=Bedrock(),
        # llm=ChatOpenAI(model="gpt-4-0125-preview", temperature=0),
        llm=claude_v2_1,
        verbose=True,
        max_iterations=10,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        agent_kwargs={
            "system_message": INTERNAL_SYSTEM_MESSAGE,
        },
    )
