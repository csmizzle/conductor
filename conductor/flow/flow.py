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
from pydantic import BaseModel, InstanceOf
from crewai import LLM
import dspy
import asyncio
from conductor.builder.agent import ResearchTeamTemplate
from conductor.flow import models, specify, runner, retriever, builders, research, team
from conductor.flow.utils import build_organization_determination_crew
from conductor.crews.rag_marketing import tools
from langchain_core.embeddings import Embeddings


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
        research_team_output = runner.run_team(self.state.specified_research_team)
        self.state.research_team_output = research_team_output
        return research_team_output


class SearchFlow(Flow):
    def __init__(
        self,
        search_team: ResearchTeamTemplate,
        organization_determination: str,
        elastic_retriever: retriever.ElasticRMClient,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.organization_determination = organization_determination
        self.search_team = search_team
        self.retriever = elastic_retriever

    @start()
    def specify_search_team(self):
        print("Specifying search team ...")
        specified_search_team = specify.specify_search_team(
            team=self.search_team, specification=self.organization_determination
        )
        return specified_search_team

    @listen(specify_search_team)
    def run_search_team(
        self, specified_search_team: models.SearchTeam
    ) -> list[runner.SearchTeamAnswers]:
        print("Running search team ...")
        search_results = runner.run_search_team(
            retriever=self.retriever, team=specified_search_team
        )
        return search_results


async def arun_flow(flow: InstanceOf[Flow]) -> str:
    return await flow.kickoff()


def run_flow(flow: InstanceOf[Flow]) -> str:
    return asyncio.run(arun_flow(flow=flow))


async def arun_search_flow(
    flow: InstanceOf[SearchFlow],
) -> list[runner.SearchTeamAnswers]:
    return await flow.kickoff()


def run_search_flow(flow: InstanceOf[SearchFlow]) -> list[runner.SearchTeamAnswers]:
    return asyncio.run(arun_search_flow(flow=flow))


class RunResult(BaseModel):
    research: list[CrewOutput]
    search: list[runner.SearchTeamAnswers]


def run_research_and_search(
    website_url: str,
    research_llm: LLM,
    research_team: ResearchTeamTemplate,
    elasticsearch: Elasticsearch,
    index_name: str,
    embeddings: InstanceOf[Embeddings],
) -> RunResult:
    """
    Executes the research and search flow for a given website URL.
    Args:
        website_url (str): The URL of the website to be researched.
        research_llm (LLM): The language model used for research.
        research_team (ResearchTeamTemplate): The template for building the research team.
        elasticsearch (Elasticsearch): The Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index.
    Returns:
        RunResult: An object containing the results of the research and search flows.
    """
    # research
    built_research_team = builders.build_team_from_template(
        team_template=research_team,
        llm=research_llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=index_name
            )
        ],
        agent_factory=research.ResearchAgentFactory,
        task_factory=research.ResearchQuestionAgentSearchTaskFactory,
        team_factory=team.ResearchTeamFactory,
    )
    research_flow = ResearchFlow(
        research_team=built_research_team,
        website_url=website_url,
        elasticsearch=elasticsearch,
        index_name=index_name,
        llm=research_llm,
    )
    research_results = run_flow(flow=research_flow)
    # search
    search_team = builders.build_search_team_from_template(team=research_team)
    search_flow = SearchFlow(
        search_team=search_team,
        organization_determination=research_flow.state.organization_determination,
        elastic_retriever=retriever.ElasticRMClient(
            elasticsearch=elasticsearch, index_name=index_name, embeddings=embeddings
        ),
    )
    answers = run_search_flow(flow=search_flow)
    return RunResult(research=research_results, search=answers)
