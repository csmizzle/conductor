"""
Search team looks against the vector database to answer the research questions
- Take search tasks and tailor them to the organization
-
"""
from pydantic import InstanceOf
import dspy
from crewai_tools import BaseTool
from crewai import LLM, Agent


class SearchAgentFactory:
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
            "agent_name: str, research_questions: list[str] -> vector_database_search_goal: str"
        )
        generated_goal = goal(
            agent_name=self.agent_name, research_questions=self.research_questions
        ).vector_database_search_goal
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
