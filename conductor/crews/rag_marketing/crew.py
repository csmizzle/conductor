from crewai import Crew, Agent, Task
from conductor.crews.rag_marketing.agents import MarketingRagAgents
from conductor.crews.rag_marketing.tasks import RagMarketingTasks
from conductor.crews.marketing.utils import task_to_task_run
from conductor.crews.models import CrewRun
from conductor.llms import claude_sonnet
from elasticsearch import Elasticsearch


class RagUrlMarketingCrew:
    """
    Start with a company URL and query a vector database for relevant information
    """

    def __init__(
        self,
        company_url: str,
        # search_query: str,
        elasticsearch: Elasticsearch,
        index_name: str,
    ) -> None:
        self.company_url = company_url
        # self.search_query = search_query
        self.elasticsearch = elasticsearch
        self.index_name = index_name

    def build_team(self) -> tuple[list[Agent], list[Task]]:
        team = []
        agents = MarketingRagAgents()
        tasks = RagMarketingTasks()
        # create all agents and add them to the team
        data_collection_agent = agents.data_collection_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        vector_search_agent = agents.vector_search_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        swot_research_agent = agents.swot_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        team.append(data_collection_agent)
        team.append(swot_research_agent)
        team.append(vector_search_agent)
        # create all tasks and add them to the team
        team_tasks = []
        url_collection_task = tasks.url_collection_task(
            agent=data_collection_agent,
            company_url=self.company_url,
        )
        # determine who the company is
        vector_search_task = tasks.vector_search_task(
            agent=vector_search_agent,
            search_query=f"Which company does this URL: {self.company_url} belong to?",
            context=[url_collection_task],
        )
        # get the company swot information
        swot_research_task = tasks.swot_research_task(
            agent=swot_research_agent,
            context=[vector_search_task],
        )
        # get the swot results from vector search
        swot_research_results = tasks.vector_multi_search_task(
            agent=vector_search_agent,
            search_query="What is a good SWOT analysis for this company?",
            context=[vector_search_task],
        )
        # get the swot company information
        team_tasks.append(url_collection_task)
        team_tasks.append(vector_search_task)
        team_tasks.append(swot_research_task)
        team_tasks.append(swot_research_results)
        return team, team_tasks

    def run(self) -> CrewRun:
        team, team_tasks = self.build_team()
        crew = Crew(
            agents=team,
            tasks=team_tasks,
        )
        result = crew.kickoff()
        # create and return crew run
        crew_run = CrewRun(
            tasks=[task_to_task_run(task) for task in crew.tasks],
            result=result,
        )
        return crew_run
