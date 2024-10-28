from crewai import Task
import dspy
import concurrent.futures
from conductor.flow import models, research


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
    return research.ResearchTeam(title=team.title, agents=team.agents, tasks=tasks)
