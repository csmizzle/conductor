"""
Redesign for agent based flow control
New things that need to happen:
- Agents needs to be able to execute research questions in parallel
- Agents needs to be able to communicate run in parallel
- Agents should be storing data then running RAG over the data
- The first implementation will be just for async collection from research questions

The first thing I will need is an crewai agent factory
"""
from crewai.flow.flow import Flow, listen, start
from crewai.crew import CrewOutput
from typing import Union
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from crewai import LLM
import dspy
import asyncio
from conductor.flow import models, specify, team
from conductor.flow.utils import build_organization_determination_crew


# configure dspy
llm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=llm)


class ResearchFlowState(BaseModel):
    organization_determination: str = ""
    research_team_output: list[CrewOutput] = []
    specified_research_team: Union[models.Team, None] = None


class ResearchFlow(Flow[ResearchFlowState]):
    """
    Flow for analyzing a company by first determining the company from their website
    - Step 1: Determine the company from the website
    - Step 2: Employ a research team to gather information about the company
    - Step 3: Analyze the information with a RAG workflow
    """

    def __init__(
        self,
        research_team: models.Team,
        website_url: str,
        elasticsearch: Elasticsearch,
        index_name: str,
        llm: LLM,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.website_url = website_url
        self.company_determination_crew = build_organization_determination_crew(
            website_url=website_url,
            elasticsearch=elasticsearch,
            index_name=index_name,
            llm=llm,
        )
        self.research_team = research_team

    @start()
    def determine_organization(self) -> str:
        """Run the determination crew to get the organization from the website

        Returns:
            str: organization determination
        """
        print("Determining organization ...")
        organization_determination = self.company_determination_crew.kickoff(
            {"website_url": self.website_url}
        )
        self.state.organization_determination = organization_determination
        return organization_determination

    @listen(determine_organization)
    def specify_research_team(self, organization_determination: str):
        print("Specifying research team ...")
        self.state.specified_research_team = specify.specify_research_team(
            team=self.research_team, specification=organization_determination
        )

    @listen(specify_research_team)
    def run_research_team(self) -> list[CrewOutput]:
        print("Running research team ...")
        research_team_output = team.run_research_team(
            self.state.specified_research_team
        )
        self.state.research_team_output = research_team_output
        return research_team_output

    # @listen(run_research_team)
    # def run_search_team(self) -> list[CrewOutput]:
    #     print("Running search team ...")
    #     search_team_output = run_search_team(self.research_team)
    #     self.state.search_team_output = search_team_output
    #     return search_team_output


async def arun_research_flow(
    research_team: models.Team,
    website_url: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    llm: LLM,
) -> str:
    flow = ResearchFlow(
        research_team=research_team,
        website_url=website_url,
        elasticsearch=elasticsearch,
        index_name=index_name,
        llm=llm,
    )
    return await flow.kickoff()


def run_research_flow(
    research_team: models.Team,
    website_url: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    llm: LLM,
) -> str:
    return asyncio.run(
        arun_research_flow(
            research_team=research_team,
            website_url=website_url,
            elasticsearch=elasticsearch,
            index_name=index_name,
            llm=llm,
        )
    )
