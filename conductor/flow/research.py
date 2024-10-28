from pydantic import BaseModel, InstanceOf
import dspy
from crewai_tools import BaseTool
from crewai import LLM, Task, Agent
from conductor.flow import signatures, models


class ResearchAgentFactory(models.AgentFactory):
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


class ResearchQuestionAgentSearchTaskFactory(models.TaskFactory):
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
            signatures.ResearchTaskExpectedOutput
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
