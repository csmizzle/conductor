"""
Test flow components
"""
from conductor.flow.models import Team
from conductor.flow.research import (
    ResearchAgentFactory,
    build_agent,
    build_agent_from_template,
    ResearchQuestionAgentSearchTaskFactory,
    build_agent_search_task,
    build_agent_search_tasks,
    build_agent_search_tasks_parallel,
    build_agents_search_tasks_parallel,
    ResearchTeamTemplate,
    ResearchTeamFactory,
    build_research_team,
    build_research_team_from_template,
)
from conductor.flow.utils import (
    build_organization_determination_crew,
)
from conductor.flow.specify import (
    TaskSpecification,
    specify_research_team,
)
from conductor.flow.flow import (
    run_research_flow,
)
from crewai import LLM, Agent, Task
from crewai.crew import CrewOutput
from conductor.builder.agent import ResearchAgentTemplate
from conductor.crews.rag_marketing import tools
from elasticsearch import Elasticsearch
import os


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


def test_build_agent_search_tasks_with_tools(elasticsearch_test_agent_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    research_questions = [
        "What is the company's mission?",
        "What are the company's values?",
    ]
    template = ResearchAgentTemplate(
        title="Company Researcher", research_questions=research_questions
    )
    llm = LLM("openai/gpt-4o")
    agent = build_agent_from_template(
        template=template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_agent_index
            )
        ],
    )
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
    assert isinstance(team, Team)
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
    assert isinstance(team, Team)
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
    assert isinstance(team, Team)
    assert all([isinstance(agent, Agent) for agent in team.agents])
    assert all([isinstance(task, Task) for task in team.tasks])


def test_organization_determination_crew(elasticsearch_test_agent_index) -> None:
    website_url = "https://www.trssllc.com"
    llm = LLM("openai/gpt-4o")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    crew = build_organization_determination_crew(
        website_url=website_url,
        llm=llm,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
    )
    result = crew.kickoff({"website_url": website_url})
    assert isinstance(result, CrewOutput)


def test_task_specification() -> None:
    title = "Company Research Team"
    agent_templates = [
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
    task = team.tasks[0]
    specifier = TaskSpecification(
        task=task, specification="The company is Thomson Reuters Special Services LLC."
    )
    specified_task = specifier.specify()
    assert isinstance(specified_task, Task)


def test_research_team_specification() -> None:
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
    research_team = build_research_team_from_template(
        team_template=team_template,
        llm=LLM("openai/gpt-4o"),
        tools=[],
    )
    specified_team = specify_research_team(
        team=research_team,
        specification="The company is Thomson Reuters Special Services LLC.",
    )
    assert isinstance(specified_team, Team)
    assert all([isinstance(agent, Agent) for agent in specified_team.agents])
    assert all([isinstance(task, Task) for task in specified_team.tasks])


def test_research_flow(elasticsearch_test_agent_index) -> None:
    title = "Company Research Team"
    website_url = "https://www.trssllc.com"
    llm = LLM("openai/gpt-4o")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    agent_templates = [
        ResearchAgentTemplate(
            title="Company Researcher",
            research_questions=[
                "What is the company's mission?",
                "What are the company's values?",
            ],
        ),
        ResearchAgentTemplate(
            title="Company Social Media Researcher",
            research_questions=[
                "What is the company's social media presence?",
                "What are the company's social media values?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(title=title, agent_templates=agent_templates)
    research_team = build_research_team_from_template(
        team_template=team_template,
        llm=llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_agent_index
            )
        ],
    )
    result = run_research_flow(
        research_team=research_team,
        website_url=website_url,
        llm=llm,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
    )
    assert isinstance(result, list)
    assert all([isinstance(output, CrewOutput) for output in result])
