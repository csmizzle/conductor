"""
Langchain Agent implementation
- Apollo Results
- Web scraping
"""
from conductor.llms import chat_bedrock
from conductor.tools import apollo_email_sender
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage
from pprint import pprint

agent_executor = chat_agent_executor.create_tool_calling_executor(
    chat_bedrock, [apollo_email_sender]
)
response = agent_executor.invoke(
    {
        "messages": [
            HumanMessage(
                content="Find me CEOs in Mclean, VA. Email the results to chrissmith700@gmail.com"
            )
        ]
    }
)
pprint(response["messages"])
