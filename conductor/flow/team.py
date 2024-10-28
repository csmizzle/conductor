from pydantic import InstanceOf
from crewai import LLM, Crew
from crewai.crew import CrewOutput
from conductor.builder import agent
from crewai_tools import BaseTool
from conductor.flow import models, research
import concurrent.futures


class ResearchTeamFactory:
    """
    Create a team of agents from a
    """

    def __init__(
        self,
        title: str,
        agent_templates: list[agent.ResearchAgentTemplate],
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
        agents = research.build_agents_from_templates_parallel(
            templates=self.agent_templates, llm=self.llm, tools=self.tools
        )
        tasks = research.build_agents_search_tasks_parallel(
            agent_templates=self.agent_templates, agents=agents
        )
        return models.Team(
            title=self.title,
            agents=agents,
            tasks=tasks,
        )


def build_research_team(
    title: str,
    agent_templates: list[agent.ResearchAgentTemplate],
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
    team_template: agent.ResearchTeamTemplate,
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
        for agent_, task in zip(self.team.agents, self.team.tasks):
            crews.append(Crew(name="research_crew", agents=[agent_], tasks=[task]))
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
