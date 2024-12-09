from crewai import Task
import dspy
import concurrent.futures
from conductor.flow import models


class DescriptionSpecification:
    """
    Specify a description
    """

    def __init__(self, name: str, description: str, specification: str) -> None:
        self.name = name
        self.description = description
        self.specification = specification

    def specify(self) -> str:
        """
        Specify the description
        """
        specifier = dspy.ChainOfThought(
            "value_name:str, description: str, specification: str -> specified_retrieval_question: str"
        )
        return specifier(
            value_name=self.name,
            description=self.description,
            specification=self.specification,
        ).specified_retrieval_question


class QuestionSpecification:
    """
    Specify a question
    """

    def __init__(self, question: str, specification: str) -> None:
        self.question = question
        self.specification = specification

    def specify(self) -> str:
        """
        Specify the question
        """
        specifier = dspy.ChainOfThought(
            "question: str, specification: str -> specified_question: str"
        )
        return specifier(
            question=self.question, specification=self.specification
        ).specified_question


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


def specify_research_team(team: models.Team, specification: str) -> models.Team:
    """
    Specify a research team
    """
    tasks = specify_tasks_parallel(tasks=team.tasks, specification=specification)
    return models.Team(title=team.title, agents=team.agents, tasks=tasks)


def specify_research_question(question: str, specification: str) -> str:
    """
    Specify a research question
    """
    return QuestionSpecification(
        question=question, specification=specification
    ).specify()


def specify_research_questions(questions: list[str], specification: str) -> list[str]:
    """
    Specify a list of research questions
    """
    specified_questions = []
    for question in questions:
        specified_questions.append(
            specify_research_question(question=question, specification=specification)
        )
    return specified_questions


def specify_research_questions_parallel(
    questions: list[str], specification: str
) -> list[str]:
    """
    Specify a list of research questions in parallel
    """
    specified_questions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for question in questions:
            futures.append(
                executor.submit(
                    specify_research_question,
                    question=question,
                    specification=specification,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            specified_questions.append(future.result())
    return specified_questions


def specify_search_agent(
    agent: models.SearchAgent, specification: str
) -> models.SearchAgent:
    """
    build a search agent
    """
    specified_questions = specify_research_questions_parallel(
        questions=agent.questions, specification=specification
    )
    return models.SearchAgent(title=agent.title, questions=specified_questions)


def specify_search_agents(
    agents: list[models.SearchAgent], specification: str
) -> list[models.SearchAgent]:
    """
    build a list of search agents
    """
    specified_agents = []
    for agent in agents:
        specified_agents.append(
            specify_search_agent(agent=agent, specification=specification)
        )
    return specified_agents


def specify_search_agents_parallel(
    agents: list[models.SearchAgent], specification: str
) -> list[models.SearchAgent]:
    """
    build a list of search agents in parallel
    """
    specified_agents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for agent in agents:
            futures.append(
                executor.submit(
                    specify_search_agent, agent=agent, specification=specification
                )
            )
        for future in concurrent.futures.as_completed(futures):
            specified_agents.append(future.result())
    return specified_agents


def specify_search_team(
    team: models.SearchTeam, specification: str
) -> models.SearchTeam:
    """
    build a search team
    """
    agents = specify_search_agents_parallel(
        agents=team.agents, specification=specification
    )
    return models.SearchTeam(title=team.title, agents=agents)


def specify_description(name: str, description: str, specification: str) -> str:
    """
    Specify a description
    """
    return DescriptionSpecification(
        name=name, description=description, specification=specification
    ).specify()


def specify_descriptions(descriptions: list[str], specification: str) -> list[str]:
    """
    Specify a list of descriptions
    """
    specified_descriptions = []
    for description in descriptions:
        specified_descriptions.append(
            specify_description(description=description, specification=specification)
        )
    return specified_descriptions


def specify_descriptions_parallel(
    descriptions: list[str], specification: str
) -> list[str]:
    """
    Specify a list of descriptions in parallel
    """
    specified_descriptions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for description in descriptions:
            futures.append(
                executor.submit(
                    specify_description,
                    description=description,
                    specification=specification,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            specified_descriptions.append(future.result())
    return specified_descriptions
