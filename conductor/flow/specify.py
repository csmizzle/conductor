import dspy
import concurrent.futures
from conductor.flow import signatures
from conductor.flow import models
from loguru import logger


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
        specifier = dspy.ChainOfThought(signatures.DescriptionSpecification)
        return specifier(
            value_name=self.name,
            description=self.description,
            specification=self.specification,
        ).specified_retrieval_question


class QuestionSpecification:
    """
    Specify a question
    """

    def __init__(
        self, question: str, specification: str, perspective: str = None
    ) -> None:
        self.question = question
        self.specification = specification
        self.perspective = perspective

    def specify(self) -> str:
        """
        Specify the question
        """
        specifier = dspy.ChainOfThought(signatures.QuestionSpecification)
        logger.info(f"Specifying question: {self.question} for {self.specification}")
        return specifier(
            question=self.question,
            specification=self.specification,
            perspective=self.perspective,
        ).specified_question


def specify_research_question(
    question: str, specification: str, perspective: str = None
) -> str:
    """
    Specify a research question
    """
    return QuestionSpecification(
        question=question, specification=specification, perspective=perspective
    ).specify()


def specify_research_questions(
    questions: list[str], specification: str, perspective: str
) -> list[str]:
    """
    Specify a list of research questions
    """
    specified_questions = []
    for question in questions:
        specified_questions.append(
            specify_research_question(
                question=question, specification=specification, perspective=perspective
            )
        )
    return specified_questions


def specify_research_questions_parallel(
    questions: list[str], specification: str, perspective: str = None
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
                    perspective=perspective,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            specified_questions.append(future.result())
    return specified_questions


def specify_search_agent(
    agent: models.SearchAgent, specification: str, perspective: str
) -> models.SearchAgent:
    """
    build a search agent
    """
    specified_questions = specify_research_questions_parallel(
        questions=agent.questions, specification=specification, perspective=perspective
    )
    return models.SearchAgent(title=agent.title, questions=specified_questions)


def specify_search_agents(
    agents: list[models.SearchAgent], specification: str, perspective: str
) -> list[models.SearchAgent]:
    """
    build a list of search agents
    """
    specified_agents = []
    for agent in agents:
        specified_agents.append(
            specify_search_agent(
                agent=agent, specification=specification, perspective=perspective
            )
        )
    return specified_agents


def specify_search_agents_parallel(
    agents: list[models.SearchAgent], specification: str, perspective: str
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
                    specify_search_agent,
                    agent=agent,
                    specification=specification,
                    perspective=perspective,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            specified_agents.append(future.result())
    return specified_agents


def specify_search_team(
    team: models.SearchTeam, specification: str, perspective: str
) -> models.SearchTeam:
    """
    build a search team
    """
    agents = specify_search_agents_parallel(
        agents=team.agents, specification=specification, perspective=perspective
    )
    return models.SearchTeam(title=team.title, perspective=perspective, agents=agents)


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
