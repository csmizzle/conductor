"""
Agent constructor function
"""
from langchain_community.llms.openai import OpenAI
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent


INTERNAL_SYSTEM_MESSAGE = """
You are a world class market researcher, can take unstructured data and create valuable insights for you users.
Always include direct links to the data you used to make your decisions.
If your tools are not giving you the right data, use your best judgement to produce the right answer.
"""


def build_internal_agent():
    return initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=[
            TavilySearchResults(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        ],
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
