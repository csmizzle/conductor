"""
General purpose builders for agent & task factories
"""
from pydantic import BaseModel, InstanceOf
from crewai_tools import BaseTool
from crewai import LLM, Agent, Task
from conductor.builder import agent
from conductor.builder.agent import ResearchAgentTemplate
from conductor.flow import models
import concurrent.futures
from tqdm import tqdm


def build_agent(
    agent_name: str,
    research_questions: list[str],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
) -> Agent:
    """
    Builds an agent using the ResearchAgentFactory
    """
    return agent_factory(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
    ).build()


def build_agent_from_template(
    template: ResearchAgentTemplate,
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
) -> Agent:
    """
    Builds an agent from a template
    """
    factory = agent_factory(
        agent_name=template.title,
        research_questions=template.research_questions,
        llm=llm,
        tools=tools,
    )
    return factory.build()


def build_agents_from_templates(
    templates: list[ResearchAgentTemplate],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
) -> list[Agent]:
    """
    Builds a list of agents from templates
    """
    agents = []
    for template in templates:
        agents.append(
            build_agent_from_template(
                agent_factory=agent_factory, template=template, llm=llm, tools=tools
            )
        )
    return agents


def build_agents_from_templates_parallel(
    templates: list[ResearchAgentTemplate],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
) -> list[Agent]:
    """
    Builds a list of agents from templates in parallel
    """
    agents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for template in templates:
            futures.append(
                executor.submit(
                    build_agent_from_template,
                    template=template,
                    llm=llm,
                    tools=tools,
                    agent_factory=agent_factory,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            agents.append(future.result())
    return agents


def build_agent_search_task(
    agent: Agent,
    research_question: str,
    task_factory: InstanceOf[models.TaskFactory],
    output_pydantic: InstanceOf[BaseModel] = None,
) -> Task:
    """
    Builds a task for the agent to search for information
    """
    return task_factory(
        agent=agent,
        research_question=research_question,
        output_pydantic=output_pydantic,
    ).build()


def build_agent_search_tasks(
    agent: Agent,
    research_questions: list[str],
    task_factory: InstanceOf[models.TaskFactory],
    output_pydantic: InstanceOf[BaseModel] = None,
) -> list[Task]:
    """
    Builds a list of tasks for the agent to search for information
    """
    tasks = []
    for research_question in tqdm(research_questions):
        tasks.append(
            build_agent_search_task(
                agent=agent,
                research_question=research_question,
                task_factory=task_factory,
                output_pydantic=output_pydantic,
            )
        )
    return tasks


def build_agent_search_tasks_parallel(
    agent: Agent,
    research_questions: list[str],
    task_factory: InstanceOf[models.TaskFactory],
    output_pydantic: InstanceOf[BaseModel] = None,
) -> list[Task]:
    """
    Builds a list of tasks for the agent to search for information in parallel
    """
    tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for research_question in research_questions:
            futures.append(
                executor.submit(
                    build_agent_search_task,
                    agent=agent,
                    research_question=research_question,
                    task_factory=task_factory,
                    output_pydantic=output_pydantic,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            tasks.append(future.result())
    return tasks


def build_agents_search_tasks_parallel(
    agent_templates: list[ResearchAgentTemplate],
    agents: list[Agent],
    task_factory: InstanceOf[models.TaskFactory],
) -> list[Task]:
    """
    Builds a list of tasks for the agents to search for information in parallel
    """
    tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for agent_template, agent_ in zip(agent_templates, agents):
            futures.append(
                executor.submit(
                    build_agent_search_tasks_parallel,
                    agent=agent_,
                    research_questions=agent_template.research_questions,
                    task_factory=task_factory,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            tasks.extend(future.result())
    return tasks


def build_research_team(
    title: str,
    agent_templates: list[agent.ResearchAgentTemplate],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
    task_factory: InstanceOf[models.TaskFactory],
    team_factory: InstanceOf[models.TeamFactory],
) -> models.Team:
    """
    Builds a research team from a list of agents
    """
    return team_factory(
        title=title,
        agent_templates=agent_templates,
        llm=llm,
        tools=tools,
        agent_factory=agent_factory,
        task_factory=task_factory,
    ).build()


def build_research_team_from_template(
    team_template: agent.ResearchTeamTemplate,
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
    agent_factory: InstanceOf[models.AgentFactory],
    task_factory: InstanceOf[models.TaskFactory],
    team_factory: InstanceOf[models.TeamFactory],
) -> models.Team:
    """
    Builds a research team from a template
    """
    return build_research_team(
        title=team_template.title,
        agent_templates=team_template.agent_templates,
        tools=tools,
        llm=llm,
        agent_factory=agent_factory,
        task_factory=task_factory,
        team_factory=team_factory,
    )
