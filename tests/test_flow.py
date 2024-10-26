from conductor.graph.flow import (
    ResearchAgentFactory,
    build_agent,
    build_agent_from_template,
    ResearchQuestionAgentSearchTaskFactory,
    build_agent_search_task,
    build_agent_search_tasks,
    build_agent_search_tasks_parallel,
    build_agents_search_tasks_parallel,
    ResearchTeamTemplate,
    ResearchTeam,
    ResearchTeamFactory,
    build_research_team,
    build_research_team_from_template,
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


def test_build_agents_search_tasks_parallel() -> None:
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(template=template, llm=llm, tools=[])
    tasks = build_agents_search_tasks_parallel(
        agents=[agent, agent], agent_templates=[template, template]
    )
    assert isinstance(tasks, list)
    assert all([isinstance(task, Task) for task in tasks])


def test_research_team_factory() -> None:
    title = "Company Research Team"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    builder = ResearchTeamFactory(
        title=title,
        agent_templates=agent_templates,
        llm=llm,
        tools=tools,
    )
    team = builder.build()
    assert isinstance(team, ResearchTeam)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_build_research_team() -> None:
    title = "Company Research Team"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    llm = LLM("openai/gpt-4o")
    tools = []
    team = build_research_team(
        title=title,
        agent_templates=agent_templates,
        llm=llm,
        tools=tools,
    )
    assert isinstance(team, ResearchTeam)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_build_research_team_from_template() -> None:
    title = "Company Research Team"
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(title=title, agent_templates=agent_templates)
    llm = LLM("openai/gpt-4o")
    tools = []
    team = build_research_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=tools,
    )
    assert isinstance(team, ResearchTeam)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])
