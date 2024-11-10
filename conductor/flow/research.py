from pydantic import BaseModel, InstanceOf
from typing import Union, Dict
import dspy
from crewai_tools import BaseTool
from crewai import LLM, Task
from crewai import Agent as BaseAgent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.utilities import Prompts
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    CrewAgentParser,
)
from conductor.flow import signatures, models


class TrimmedPromptTask(Task):
    def prompt(self) -> str:
        tasks_slices = [self.description]

        output = self.i18n.slice("expected_output").format(
            expected_output=self.expected_output
        )
        tasks_slices = [self.description, output]
        prompt = "\n".join(tasks_slices).strip()
        return prompt


class TrimmedCrewAgentExecutor(CrewAgentExecutor):
    def _format_answer(self, answer: str) -> Union[AgentAction, AgentFinish]:
        return CrewAgentParser(agent=self.agent).parse(
            answer.strip()
        )  # stripping for AWS

    def _format_msg(self, prompt: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": prompt.strip()}  # stripping for AWS


class Agent(BaseAgent):
    def create_agent_executor(self, tools=None, task=None) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        tools = tools or self.tools or []
        parsed_tools = self._parse_tools(tools)

        prompt = Prompts(
            agent=self,
            tools=tools,
            i18n=self.i18n,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        self.agent_executor = TrimmedCrewAgentExecutor(
            llm=self.llm,
            task=task,
            agent=self,
            crew=self.crew,
            tools=parsed_tools,
            prompt=prompt,
            original_tools=tools,
            stop_words=stop_words,
            max_iter=self.max_iter,
            tools_handler=self.tools_handler,
            tools_names=self.__tools_names(parsed_tools),
            tools_description=self._render_text_description_and_args(parsed_tools),
            step_callback=self.step_callback,
            function_calling_llm=self.function_calling_llm,
            respect_context_window=self.respect_context_window,
            request_within_rpm_limit=(
                self._rpm_controller.check_or_wait if self._rpm_controller else None
            ),
            callbacks=[TokenCalcHandler(self._token_process)],
        )


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
        max_iter: int = 5,
    ) -> None:
        self.agent_name = agent_name
        self.research_questions = research_questions
        self.tools = tools
        self.llm = llm
        self.max_iter = max_iter

    def _build_backstory(self) -> str:
        backstory = dspy.Predict(signatures.AgentBackstory)
        return backstory(
            agent_name=self.agent_name, research_questions=self.research_questions
        ).backstory

    def _build_goal(self) -> str:
        goal = dspy.Predict(signatures.AgentSearchEngineResearchGoal)
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
            max_iter=self.max_iter,
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

    def build(self) -> TrimmedPromptTask:
        """
        Builds a task for the agent to search for information
        """
        task_description = self._build_description()
        return TrimmedPromptTask(
            description=task_description,
            agent=self.agent,
            output_pydantic=self.output_pydantic,
            expected_output="A simple confirmation of the data the agent collected",
        )
