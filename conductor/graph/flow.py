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
import concurrent.futures
from crewai import LLM
import dspy
from tqdm import tqdm


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


def build_agents_from_templates(
    templates: list[ResearchAgentTemplate], llm: LLM, tools: list
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
    templates: list[ResearchAgentTemplate], llm: LLM, tools: list
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
            "agent_role: str, agent_research_question: str, agent_goal: str, agent_backstory: str, task_description: str -> search_engine_expected_output: str"
        )
        return expected_output(
            agent_role=self.agent.role,
            agent_research_question=self.research_question,
            agent_goal=self.agent.goal,
            agent_backstory=self.agent.backstory,
            task_description=task_description,
        ).search_engine_expected_output

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
        tools: list,
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
    title: str, agent_templates: list[ResearchAgentTemplate], llm: LLM, tools: list
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
    tools: list,
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
