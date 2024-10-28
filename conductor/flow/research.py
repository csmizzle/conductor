from pydantic import BaseModel, InstanceOf
import dspy
import concurrent.futures
from tqdm import tqdm
from crewai_tools import BaseTool
from crewai import LLM, Task, Agent, Crew
from crewai.crew import CrewOutput
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.flow import signatures, models


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
            "agent_name: str, research_questions: list[str] -> search_engine_research_goal: str"
        )
        generated_goal = goal(
            agent_name=self.agent_name, research_questions=self.research_questions
        ).search_engine_research_goal
        return generated_goal

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
        description = dspy.ChainOfThought(signatures.ResearchTaskDescription)
        return description(
            agent_role=self.agent.role,
            agent_research_question=self.research_question,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
        ).task_description

    def _build_expected_output(self, task_description: str) -> str:
        generate_expected_output = dspy.ChainOfThought(
            signatures.ResearchAgentExpectedOutput
        )
        return generate_expected_output(
            agent_role=self.agent.role,
            agent_research_question=self.research_question,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
            task_description=task_description,
        ).expected_output

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

    def build(self) -> models.Team:
        """
        Build a research team from a list of agents
        """
        agents = build_agents_from_templates_parallel(
            templates=self.agent_templates, llm=self.llm, tools=self.tools
        )
        tasks = build_agents_search_tasks_parallel(
            agent_templates=self.agent_templates, agents=agents
        )
        return models.Team(
            title=self.title,
            agents=agents,
            tasks=tasks,
        )


def build_research_team(
    title: str,
    agent_templates: list[ResearchAgentTemplate],
    llm: LLM,
    tools: list[InstanceOf[BaseTool]],
) -> models.Team:
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
) -> models.Team:
    """
    Builds a research team from a template
    """
    return build_research_team(
        title=team_template.title,
        agent_templates=team_template.agent_templates,
        tools=tools,
        llm=llm,
    )


class ResearchTeamRunner:
    """`
    Run a research team by executing the tasks in parallel
    """

    def __init__(self, team: models.Team) -> None:
        self.team = team
        self.crews: list[Crew] = self._assemble_crews()

    @staticmethod
    def _run_research_crew(crew: Crew) -> None:
        print(f"Running crew {crew.id} ...")
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


def run_research_team(team: models.Team) -> list[CrewOutput]:
    """
    Run a research team by executing the tasks in parallel
    """
    return ResearchTeamRunner(team=team).run()
