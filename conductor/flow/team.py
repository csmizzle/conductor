from pydantic import InstanceOf
from crewai import LLM
from conductor.builder import agent
from crewai_tools import BaseTool
from conductor.flow import models, builders


class ResearchTeamFactory(models.TeamFactory):
    """
    Create a team of agents from a
    """

    def __init__(
        self,
        title: str,
        agent_templates: list[agent.ResearchAgentTemplate],
        agent_factory: InstanceOf[models.AgentFactory],
        task_factory: InstanceOf[models.TaskFactory],
        llm: LLM,
        tools: list[InstanceOf[BaseTool]],
        max_iter: int = 1,
    ) -> None:
        self.title = title
        self.agent_templates = agent_templates
        self.llm = llm
        self.tools = tools
        self.agent_factory = agent_factory
        self.task_factory = task_factory
        self.max_iter = max_iter

    def build(self) -> models.Team:
        """
        Build a research team from a list of agents
        """
        agents = builders.build_agents_from_templates_parallel(
            templates=self.agent_templates,
            llm=self.llm,
            tools=self.tools,
            agent_factory=self.agent_factory,
            max_iter=self.max_iter,
        )
        tasks = builders.build_agents_search_tasks_parallel(
            agent_templates=self.agent_templates,
            agents=agents,
            task_factory=self.task_factory,
        )
        return models.Team(
            title=self.title,
            agents=agents,
            tasks=tasks,
        )
