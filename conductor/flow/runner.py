from conductor.flow import models
from conductor.flow.rag import CitedAnswerWithCredibility, CitationRAG
from conductor.flow.retriever import ElasticRMClient
from pydantic import BaseModel
from crewai import Crew
from crewai.crew import CrewOutput
import concurrent.futures


class TeamRunner:
    """
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


class SearchTeamAnswers(BaseModel):
    agent_title: str
    answers: list[CitedAnswerWithCredibility]


class SearchTeamRunner:
    def __init__(self, team: models.SearchTeam, retriever: ElasticRMClient) -> None:
        self.team = team
        self.retriever = CitationRAG(elastic_retriever=retriever)

    def _run_search_agent_question(self, question: str) -> CitedAnswerWithCredibility:
        return self.retriever(question=question)

    def _run_search_agent_parallel(
        self, agent: models.SearchAgent
    ) -> SearchTeamAnswers:
        """
        Run research questions in parallel
        """
        answers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for question in agent.questions:
                futures.append(
                    executor.submit(self._run_search_agent_question, question)
                )
            for future in concurrent.futures.as_completed(futures):
                answers.append(future.result())
        return SearchTeamAnswers(agent_title=agent.title, answers=answers)

    def _run_agents_parallel(self) -> list[SearchTeamAnswers]:
        """
        Run agents in parallel
        """
        answers = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for agent in self.team.agents:
                futures.append(executor.submit(self._run_search_agent_parallel, agent))
            for future in concurrent.futures.as_completed(futures):
                answers.append(future.result())
        return answers

    def run(self) -> list[SearchTeamAnswers]:
        answers = self._run_agents_parallel()
        return answers


def run_team(team: models.Team) -> list[CrewOutput]:
    """
    Run a research team by executing the tasks in parallel
    """
    return TeamRunner(team=team).run()


def run_search_team(
    team: models.SearchTeam, retriever: ElasticRMClient
) -> list[SearchTeamAnswers]:
    return SearchTeamRunner(team=team, retriever=retriever).run()
