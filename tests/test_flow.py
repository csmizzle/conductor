from conductor.graph.flow import (
    ResearchAgentFactory,
    build_agent,
    build_agent_from_template,
)
from crewai import LLM
from crewai import Agent
from conductor.builder.agent import ResearchAgentTemplate


def test_research_agent_factory():
    agent_name = "Company Researcher"
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    builder = ResearchAgentFactory(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
    )
    agent = builder.build()
    assert isinstance(agent, Agent)


def test_build_agent():
    agent_name = "Company Researcher"
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    agent = build_agent(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
    )
    assert isinstance(agent, Agent)


def test_build_agent_from_template():
    template = ResearchAgentTemplate(
        title="Company Researcher",
        research_questions=[
            "What is the company's mission?",
            "What are the company's values?",
        ],
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    assert isinstance(agent, Agent)
