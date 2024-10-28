from conductor.flow import models
from crewai import Crew
from crewai.crew import CrewOutput
import concurrent.futures


class TeamRunner:
    """`
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


def run_team(team: models.Team) -> list[CrewOutput]:
    """
    Run a research team by executing the tasks in parallel
    """
    return TeamRunner(team=team).run()
