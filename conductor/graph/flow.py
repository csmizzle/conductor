"""
Redesign for agent based flow control
New things that need to happen:
- Agents needs to be able to execute research questions in parallel
- Agents needs to be able to communicate run in parallel
- Agents should be storing data then running RAG over the data
- The first implementation will be just for async collection from research questions

The first thing I will need is an crewai agent factory
"""
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from crewai.flow.flow import Flow, listen, start
from conductor.crews.rag_marketing import tools
from crewai.agent import Agent
from crewai.crew import CrewOutput
from crewai_tools.tools.base_tool import BaseTool
from crewai import Task, Crew
from pydantic import BaseModel, InstanceOf
from typing import Union
from elasticsearch import Elasticsearch
import concurrent.futures
from crewai import LLM
import dspy
from tqdm import tqdm
import asyncio


# configure dspy
llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)

# configure crew llm
crew_llm = LLM("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")


class ResearchTeam(BaseModel):
    title: str
    agents: list[Agent]
    tasks: list[Task]


class ResearchAgentFactory:
    """
    Factory class for creating agents
    """

    def __init__(
        self,
        agent_name: str,
        research_questions: list[str],
        llm: LLM,
        tools: list[InstanceOf[BaseTool]],
    ) -> None:
        self.agent_name = agent_name
        self.research_questions = research_questions
        self.tools = tools
        self.llm = llm

    def _build_backstory(self) -> str:
        backstory = dspy.ChainOfThought(
            "agent_name: str, research_questions: list[str] -> backstory: str"
        )
        return backstory(
            agent_name=self.agent_name, research_questions=self.research_questions
        ).backstory

    def _build_goal(self) -> str:
        goal = dspy.ChainOfThought(
            "agent_name: str, research_questions: list[str] -> goal: str"
        )
        return goal(
            agent_name=self.agent_name, research_questions=self.research_questions
        ).goal

    def build(self) -> Agent:
        """
        Builds the agent using dspy deriving goal & backstory from a combination of research questions and agent name
        """
        return Agent(
            role=self.agent_name,
            goal=self._build_goal(),
            backstory=self._build_backstory(),
            tools=self.tools,
            llm=self.llm,
        )


def build_agent(
    agent_name: str,
    research_questions: list[str],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
) -> Agent:
    """
    Builds an agent using the ResearchAgentFactory
    """
    return ResearchAgentFactory(
        agent_name=agent_name,
        research_questions=research_questions,
        llm=llm,
        tools=tools,
    ).build()


def build_agent_from_template(
    template: ResearchAgentTemplate, llm: LLM, tools: list[InstanceOf[BaseTool]]
) -> Agent:
    """
    Builds an agent from a template
    """
    factory = ResearchAgentFactory(
        agent_name=template.title,
        research_questions=template.research_questions,
        llm=llm,
        tools=tools,
    )
    return factory.build()


def build_agents_from_templates(
    templates: list[ResearchAgentTemplate], llm: LLM, tools: list[InstanceOf[BaseTool]]
) -> list[Agent]:
    """
    Builds a list of agents from templates
    """
    agents = []
    for template in templates:
        agents.append(
            build_agent_from_template(template=template, llm=llm, tools=tools)
        )
    return agents


def build_agents_from_templates_parallel(
    templates: list[ResearchAgentTemplate], llm: LLM, tools: list[InstanceOf[BaseTool]]
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
                )
            )
        for future in concurrent.futures.as_completed(futures):
            agents.append(future.result())
    return agents


# task builder
class ResearchQuestionAgentSearchTaskFactory:
    """
    Agent task factory that create
    """

    def __init__(
        self,
        agent: Agent,
        research_question: str,
        output_pydantic: InstanceOf[BaseModel] = None,
    ) -> None:
        self.agent = agent
        self.research_question = research_question
        self.output_pydantic = output_pydantic

    def _build_description(self) -> str:
        description = dspy.ChainOfThought(
            "agent_role: str, agent_research_question: str, agent_goal: str, agent_backstory: str -> search_engine_task_description: str"
        )
        return description(
            agent_role=self.agent.role,
            agent_research_question=self.research_question,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
        ).search_engine_task_description

    def _build_expected_output(self, task_description: str) -> str:
        expected_output = dspy.ChainOfThought(
            "agent_role: str, agent_research_question: str, agent_goal: str, agent_backstory: str, task_description: str -> ingested_documents_expected_output: str"
        )
        return expected_output(
            agent_role=self.agent.role,
            agent_research_question=self.research_question,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
            task_description=task_description,
        ).ingested_documents_expected_output

    def build(self) -> Task:
        """
        Builds a task for the agent to search for information
        """
        task_description = self._build_description()
        expected_output = self._build_expected_output(task_description)
        return Task(
            description=task_description,
            agent=self.agent,
            output_pydantic=self.output_pydantic,
            expected_output=expected_output,
        )


def build_agent_search_task(
    agent: Agent, research_question: str, output_pydantic: InstanceOf[BaseModel] = None
) -> Task:
    """
    Builds a task for the agent to search for information
    """
    return ResearchQuestionAgentSearchTaskFactory(
        agent=agent,
        research_question=research_question,
        output_pydantic=output_pydantic,
    ).build()


def build_agent_search_tasks(
    agent: Agent,
    research_questions: list[str],
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
                output_pydantic=output_pydantic,
            )
        )
    return tasks


def build_agent_search_tasks_parallel(
    agent: Agent,
    research_questions: list[str],
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
                    output_pydantic=output_pydantic,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            tasks.append(future.result())
    return tasks


def build_agents_search_tasks_parallel(
    agent_templates: list[ResearchAgentTemplate],
    agents: list[Agent],
) -> list[Task]:
    """
    Builds a list of tasks for the agents to search for information in parallel
    """
    tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for agent_template, agent in zip(agent_templates, agents):
            futures.append(
                executor.submit(
                    build_agent_search_tasks_parallel,
                    agent=agent,
                    research_questions=agent_template.research_questions,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            tasks.extend(future.result())
    return tasks


class ResearchTeamFactory:
    """
    Create a team of agents from a
    """

    def __init__(
        self,
        title: str,
        agent_templates: list[ResearchAgentTemplate],
        llm: LLM,
        tools: list[InstanceOf[BaseTool]],
    ) -> None:
        self.title = title
        self.agent_templates = agent_templates
        self.llm = llm
        self.tools = tools

    def build(self) -> ResearchTeam:
        """
        Build a research team from a list of agents
        """
        agents = build_agents_from_templates_parallel(
            templates=self.agent_templates, llm=self.llm, tools=self.tools
        )
        tasks = build_agents_search_tasks_parallel(
            agent_templates=self.agent_templates, agents=agents
        )
        return ResearchTeam(
            title=self.title,
            agents=agents,
            tasks=tasks,
        )


def build_research_team(
    title: str,
    agent_templates: list[ResearchAgentTemplate],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
) -> ResearchTeam:
    """
    Builds a research team from a list of agents
    """
    return ResearchTeamFactory(
        title=title,
        agent_templates=agent_templates,
        llm=llm,
        tools=tools,
    ).build()


def build_research_team_from_template(
    team_template: ResearchTeamTemplate,
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
) -> ResearchTeam:
    """
    Builds a research team from a template
    """
    return build_research_team(
        title=team_template.title,
        agent_templates=team_template.agent_templates,
        tools=tools,
        llm=llm,
    )


def build_organization_determination_crew(
    website_url: str, elasticsearch: Elasticsearch, index_name: str, llm: LLM
):
    organization_determination_agent = Agent(
        role="Organization Determination Agent",
        goal="Determine the organization from the website",
        backstory="The agent is tasked with determining the organization from the website.",
        tools=[
            tools.ScrapeWebsiteWithContentIngestTool(
                elasticsearch=elasticsearch,
                index_name=index_name,
            )
        ],
        allow_delegation=False,
        llm=llm,
    )
    organization_determination_task = Task(
        description=f"Determine the organization from the website {website_url}",
        agent=organization_determination_agent,
        expected_output="The determined organization from the website content with a sentence on reasoning.",
    )
    company_determination_crew = Crew(
        name="company_determination_crew",
        agents=[organization_determination_agent],
        tasks=[organization_determination_task],
    )
    return company_determination_crew


class TaskSpecification:
    def __init__(self, task: Task, specification: str) -> None:
        self.task = task
        self.specification = specification

    def _specify_description(self) -> str:
        specifier = dspy.ChainOfThought(
            "task_description: str, specification: str -> specified_task_description: str"
        )
        return specifier(
            task_description=self.task.description, specification=self.specification
        ).specified_task_description

    def _specify_expected_output(self) -> str:
        specifier = dspy.ChainOfThought(
            "task_description: str, specification: str -> specified_expected_output: str"
        )
        return specifier(
            task_description=self.task.description, specification=self.specification
        ).specified_expected_output

    def specify(self) -> Task:
        """
        Specify the task
        """
        specified_description = self._specify_description()
        specified_expected_output = self._specify_expected_output()
        return Task(
            description=specified_description,
            agent=self.task.agent,
            expected_output=specified_expected_output,
            output_pydantic=self.task.output_pydantic,
        )


def specify_task(task: Task, specification: str) -> Task:
    """
    Specify a task
    """
    return TaskSpecification(task=task, specification=specification).specify()


def specify_tasks(tasks: list[Task], specification: str) -> list[Task]:
    """
    Specify a list of tasks
    """
    specified_tasks = []
    for task in tasks:
        specified_tasks.append(specify_task(task=task, specification=specification))
    return specified_tasks


def specify_tasks_parallel(tasks: list[Task], specification: str) -> list[Task]:
    """
    Specify a list of tasks in parallel
    """
    specified_tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(specify_task, task=task, specification=specification)
            )
        for future in concurrent.futures.as_completed(futures):
            specified_tasks.append(future.result())
    return specified_tasks


def specify_research_team(team: ResearchTeam, specification: str) -> ResearchTeam:
    """
    Specify a research team
    """
    tasks = specify_tasks_parallel(tasks=team.tasks, specification=specification)
    return ResearchTeam(title=team.title, agents=team.agents, tasks=tasks)


class ResearchTeamRunner:
    """`
    Run a research team by executing the tasks in parallel
    """

    def __init__(self, team: ResearchTeam) -> None:
        self.team = team
        self.crews: list[Crew] = self._assemble_crews()

    @staticmethod
    def _run_research_crew(crew: Crew) -> None:
        return crew.kickoff()

    def _assemble_crews(self) -> None:
        crews = []
        for agent, task in zip(self.team.agents, self.team.tasks):
            crews.append(Crew(name="research_crew", agents=[agent], tasks=[task]))
        return crews

    def run(self) -> list[CrewOutput]:
        """Run the assembled teams in parallel

        Returns:
            outputs: the crew outputs
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for crew in self.crews:
                futures.append(executor.submit(self._run_research_crew, crew))
            return [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]


def run_research_team(team: ResearchTeam) -> list[CrewOutput]:
    """
    Run a research team by executing the tasks in parallel
    """
    return ResearchTeamRunner(team=team).run()


class ResearchFlow(Flow):
    """
    Flow for analyzing a company by first determining the company from their website
    - Step 1: Determine the company from the website
    - Step 2: Employ a research team to gather information about the company
    - Step 3: Analyze the information with a RAG workflow
    """

    def __init__(
        self,
        research_team: ResearchTeam,
        website_url: str,
        elasticsearch: Elasticsearch,
        index_name: str,
        llm: LLM,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.website_url = website_url
        self.company_determination_crew = build_organization_determination_crew(
            website_url=website_url,
            elasticsearch=elasticsearch,
            index_name=index_name,
            llm=llm,
        )
        self.research_team = research_team
        self.specified_research_team: Union[ResearchTeam, None] = None

    @start()
    def determine_organization(self) -> str:
        """Run the determination crew to get the organization from the website

        Returns:
            str: organization determination
        """
        return self.company_determination_crew.kickoff(
            {"website_url": self.website_url}
        )

    @listen(determine_organization)
    def specify_research_team(self, organization_determination: str):
        self.specified_research_team = specify_research_team(
            team=self.research_team, specification=organization_determination
        )

    @listen(specify_research_team)
    def run_research_team(self) -> list[CrewOutput]:
        return run_research_team(self.specified_research_team)


async def arun_research_flow(
    research_team: ResearchTeam,
    website_url: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    llm: LLM,
) -> str:
    flow = ResearchFlow(
        research_team=research_team,
        website_url=website_url,
        elasticsearch=elasticsearch,
        index_name=index_name,
        llm=llm,
    )
    return await flow.kickoff()


def run_research_flow(
    research_team: ResearchTeam,
    website_url: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    llm: LLM,
) -> str:
    return asyncio.run(
        arun_research_flow(
            research_team=research_team,
            website_url=website_url,
            elasticsearch=elasticsearch,
            index_name=index_name,
            llm=llm,
        )
    )
