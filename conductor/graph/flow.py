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
from crewai.agent import Agent
from crewai import Task
from pydantic import BaseModel, InstanceOf
from crewai import LLM
import dspy

# configure dspy
llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)

# configure crew llm
crew_llm = LLM("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")


class ResearchTeam(BaseModel):
    title: str
    agents: list[Agent]


class ResearchAgentFactory:
    """
    Factory class for creating agents
    """

    def __init__(
        self, agent_name: str, research_questions: list[str], llm: LLM, tools: list
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
    agent_name: str, research_questions: list[str], llm: LLM, tools: list
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
    template: ResearchAgentTemplate, llm: LLM, tools: list
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


# task builder
class ResearchAgentSearchTaskFactory:
    """
    Agent task factory that create
    """

    def __init__(
        self,
        agent: Agent,
        research_questions: list[str],
        expected_output: InstanceOf[BaseModel],
    ) -> None:
        self.agent = agent
        self.research_questions = research_questions
        self.expected_output = expected_output

    def _build_description(self) -> str:
        description = dspy.ChainOfThought(
            "agent_role: str, agent_research_questions: list[str], agent_goal: str, agent_backstory: str -> task_description: str"
        )
        return description(
            agent_role=self.agent.role,
            agent_research_questions=self.research_questions,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
        ).task_description

    def build(self) -> Task:
        """
        Builds a task for the agent to search for information
        """
        return Task(
            description=self._build_description(),
            agent=self.agent,
            expected_output=self.expected_output,
        )


def build_agent_search_task(
    agent: Agent, research_questions: list[str], expected_output: InstanceOf[BaseModel]
) -> Task:
    """
    Builds a task for the agent to search for information
    """
    return ResearchAgentSearchTaskFactory(
        agent=agent,
        research_questions=research_questions,
        expected_output=expected_output,
    ).build()


class ResearchTeamFactory:
    """
    Create a team of agents from a
    """

    def __init__(self, team_name: str, agents: list[ResearchAgentTemplate]) -> None:
        self.team_name = team_name
        self.agents = agents

    def build(self) -> ResearchTeam:
        """
        Build a research team from a list of agents
        """
        agents = []
        for agent in self.agents:
            agents.append(build_agent_from_template(agent))
        return ResearchTeam(team_name=self.team_name, agents=agents)


def build_research_team(
    team_name: str, agents: list[ResearchAgentTemplate]
) -> ResearchTeam:
    """
    Builds a research team from a list of agents
    """
    return ResearchTeamFactory(team_name=team_name, agents=agents).build()


def build_research_team_from_template(template: ResearchTeamTemplate) -> ResearchTeam:
    """
    Builds a research team from a template
    """
    return build_research_team(template.team_name, template.agents)
