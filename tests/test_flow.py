from conductor.graph.flow import (
    ResearchAgentFactory,
    build_agent,
    build_agent_from_template,
    ResearchQuestionAgentSearchTaskFactory,
    build_agent_search_task,
    build_agent_search_tasks,
    build_agent_search_tasks_parallel,
)
from crewai import LLM, Agent, Task
from conductor.builder.agent import ResearchAgentTemplate


def test_research_agent_factory() -> None:
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


def test_build_agent() -> None:
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


def test_build_agent_from_template() -> None:
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


def test_build_agent_task_factory() -> None:
    # build agent
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    # build agent search task
    task_builder = ResearchQuestionAgentSearchTaskFactory(
        agent=agent, research_question=research_questions[0]
    )
    task = task_builder.build()
    assert isinstance(task, Task)


def test_build_agent_search_task() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    task = build_agent_search_task(agent=agent, research_question=research_questions[0])
    assert isinstance(task, Task)


def test_build_agent_search_tasks() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    tasks = build_agent_search_tasks(agent=agent, research_questions=research_questions)
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_build_agent_search_tasks_parallel() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    tasks = build_agent_search_tasks_parallel(
        agent=agent, research_questions=research_questions
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])
