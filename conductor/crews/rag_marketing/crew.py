from crewai import Crew, Agent, Task
from conductor.crews.rag_marketing.agents import MarketingRagAgents
from conductor.crews.rag_marketing.tasks import RagMarketingTasks
from conductor.crews.marketing.utils import task_to_task_run
from conductor.crews.models import CrewRun
from conductor.crews.cache import RedisCrewCacheHandler
from conductor.crews.handlers import RedisCacheHandlerCrew
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.crew import CrewOutput
from crewai.llm import LLM
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import Callable
import asyncio


claude_sonnet = LLM(model="anthropic.claude-3-sonnet-20240229-v1:0")


class TeamTaskAssignment(BaseModel):
    team: list[Agent]
    tasks: list[Task]


class SearchCrew(BaseModel):
    company_structure: Crew
    personnel: Crew
    swot: Crew
    competitors: Crew
    company_history: Crew
    pricing: Crew
    recent_events: Crew
    products_services: Crew
    market: Crew


class RagUrlMarketingCrew:
    """
    Start with a company URL and query a vector database for relevant information
    """

    def __init__(
        self,
        url: str,
        # search_query: str,
        elasticsearch: Elasticsearch,
        index_name: str,
        cache: bool = False,
        redis: bool = False,
        task_callback: Callable = None,
    ) -> None:
        self.url = url
        # self.search_query = search_query
        self.tasks = RagMarketingTasks()
        self.agents = MarketingRagAgents()
        self.elasticsearch = elasticsearch
        self.index_name = index_name
        self.cache = cache
        if redis:
            self.cache_handler = RedisCrewCacheHandler()
        else:
            self.cache_handler = CacheHandler()
        self.task_callback = task_callback
        self.company_determination_run = None

    def _build_determination_task(self) -> TeamTaskAssignment:
        company_determination_team = []
        team_tasks = []
        # search by metadata string
        vector_meta_search_agent = self.agents.vector_search_metadata_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        # create the company determination task
        data_collection_agent = self.agents.data_collection_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        company_determination_team.append(data_collection_agent)
        url_collection_task = self.tasks.url_collection_task(
            name="URL Collection Research",
            agent=data_collection_agent,
            company_url=self.url,
        )
        team_tasks.append(url_collection_task)
        company_determination_search_task = self.tasks.vector_metadata_search_task(
            name="Company Determination Research",
            agent=vector_meta_search_agent,
            url=self.url,
        )
        team_tasks.append(company_determination_search_task)
        return TeamTaskAssignment(
            team=company_determination_team,
            tasks=team_tasks,
        )

    def run_team_task(self, teak_task: TeamTaskAssignment) -> CrewOutput:
        if self.cache:
            crew = RedisCacheHandlerCrew(
                agents=teak_task.team,
                tasks=teak_task.tasks,
                cache=self.cache,
                _cache_handler=self.cache_handler,
                task_callback=self.task_callback,
            )
        else:
            crew = Crew(
                agents=teak_task.team,
                tasks=teak_task.tasks,
                task_callback=self.task_callback,
            )
        result = crew.kickoff()
        return result

    def _run_company_determination(self) -> CrewOutput:
        return self.run_team_task(self._build_determination_task)

    def build_search_task(self, company_determination_search_task: Task) -> list[Crew]:
        # company structure
        company_structure_research_agent = self.agents.company_structure_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        company_structure_research_task = self.tasks.company_structure_research_task(
            name="Company Structure Research",
            agent=company_structure_research_agent,
            context=[company_determination_search_task],
        )
        # personnel
        personnel_research_agent = self.agents.personnel_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        personnel_research_task = self.tasks.personnel_research_task(
            name="Personnel Research",
            agent=personnel_research_agent,
            context=[company_determination_search_task],
        )
        # SWOT analysis
        swot_research_agent = self.agents.swot_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        swot_research_task = self.tasks.swot_research_task(
            name="SWOT Research",
            agent=swot_research_agent,
            context=[
                company_determination_search_task,
            ],
        )
        # competitors task
        competitors_research_agent = self.agents.competitor_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        competitors_research_task = self.tasks.competitor_research_task(
            name="Competitor Research",
            agent=competitors_research_agent,
            context=[
                company_determination_search_task,
                personnel_research_task,
                company_structure_research_task,
            ],
        )
        # company history
        company_history_research_agent = self.agents.company_history_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        company_history_research_task = self.tasks.company_history_research_task(
            name="Company History Research",
            agent=company_history_research_agent,
            context=[company_determination_search_task],
        )
        # pricing research
        pricing_research_agent = self.agents.pricing_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        pricing_research_task = self.tasks.pricing_research_task(
            name="Pricing Research",
            agent=pricing_research_agent,
            context=[company_determination_search_task],
        )
        # recent events
        recent_events_research_agent = self.agents.recent_events_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        recent_events_research_task = self.tasks.recent_events_research_task(
            name="Recent Events Research",
            agent=recent_events_research_agent,
            context=[company_determination_search_task],
        )
        # products and services
        products_services_research_agent = (
            self.agents.product_and_services_research_agent(
                elasticsearch=self.elasticsearch,
                index_name=self.index_name,
                llm=claude_sonnet,
            )
        )
        products_services_research_task = (
            self.tasks.products_and_services_research_task(
                name="Products and Service Research",
                agent=products_services_research_agent,
                context=[company_determination_search_task],
            )
        )
        # market research
        market_research_agent = self.agents.market_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        market_research_task = self.tasks.market_analysis_research_task(
            name="Market Research",
            agent=market_research_agent,
            context=[company_determination_search_task],
        )
        return [
            Crew(
                agents=[company_structure_research_agent],
                tasks=[company_structure_research_task],
            ),
            Crew(agents=[personnel_research_agent], tasks=[personnel_research_task]),
            Crew(agents=[swot_research_agent], tasks=[swot_research_task]),
            Crew(
                agents=[competitors_research_agent], tasks=[competitors_research_task]
            ),
            Crew(
                agents=[company_history_research_agent],
                tasks=[company_history_research_task],
            ),
            Crew(agents=[pricing_research_agent], tasks=[pricing_research_task]),
            Crew(
                agents=[recent_events_research_agent],
                tasks=[recent_events_research_task],
            ),
            Crew(
                agents=[products_services_research_agent],
                tasks=[products_services_research_task],
            ),
            Crew(agents=[market_research_agent], tasks=[market_research_task]),
        ]

    async def _execute_async_crews(self, search_crew: SearchCrew) -> list[CrewOutput]:
        # kick off all search tasks
        company_structure_result = search_crew.company_structure.kickoff_async()
        personnel_result = search_crew.personnel.kickoff_async()
        swot_result = search_crew.swot.kickoff_async()
        competitors_result = search_crew.competitors.kickoff_async()
        company_history_result = search_crew.company_history.kickoff_async()
        pricing_result = search_crew.pricing.kickoff_async()
        recent_events_result = search_crew.recent_events.kickoff_async()
        products_services_result = search_crew.products_services.kickoff_async()
        market_result = search_crew.market.kickoff_async()
        # wait for all results
        results = await asyncio.gather(
            company_structure_result,
            personnel_result,
            swot_result,
            competitors_result,
            company_history_result,
            pricing_result,
            recent_events_result,
            products_services_result,
            market_result,
        )
        return results

    def run_search_task(self, search_crew: SearchCrew) -> list[CrewOutput]:
        results = asyncio.run(self._execute_async_crews(search_crew))
        return results

    def build_research_task(
        self, company_determination_search_task: Task
    ) -> TeamTaskAssignment:
        # Question tasks
        # get the company structure results
        team = []
        team_tasks = []
        vector_search_agent = self.agents.vector_search_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        company_structure_research_results = self.tasks.vector_multi_search_task(
            name="Company Structure Search",
            agent=vector_search_agent,
            search_query="What is the structure of this company? Include things like beneficial owners, subsidiaries, and parent companies.",
            context=[company_determination_search_task],
            section_name="Company Structure",
        )
        team_tasks.append(company_structure_research_results)
        # get the company history results
        company_history_research_results = self.tasks.vector_multi_search_task(
            name="Company History Search",
            agent=vector_search_agent,
            search_query="What is the history of this company?",
            instructions="Include founding date, key events, mergers and acquisitions, and any notable acquisitions.",
            context=[
                company_determination_search_task,
            ],
            section_name="Company History",
        )
        team_tasks.append(company_history_research_results)
        # get the personnel results
        personnel_research_results = self.tasks.vector_multi_search_task(
            name="Personnel Search",
            agent=vector_search_agent,
            search_query="Who are the executives of this company?",
            instructions="Collect any contact information available and include a short bio for each executive if available.",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                company_history_research_results,
            ],
            section_name="Personnel",
        )
        team_tasks.append(personnel_research_results)
        # get the competitors results
        competitors_research_results = self.tasks.vector_multi_search_task(
            name="Competitors Search",
            agent=vector_search_agent,
            search_query="Who are the competitors of this company",
            instructions="Include a high, medium, or low risk analysis with a short analysis for each competitor.",
            context=[
                company_determination_search_task,
                company_history_research_results,
                personnel_research_results,
                company_structure_research_results,
            ],
            section_name="Competitors",
        )
        team_tasks.append(competitors_research_results)
        # get the pricing results
        pricing_research_results = self.tasks.vector_multi_search_task(
            name="Pricing Search",
            agent=vector_search_agent,
            search_query="What is the pricing information for this company?",
            instructions=" Include any pricing models or pricing strategies.",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                personnel_research_results,
                competitors_research_results,
                company_history_research_results,
            ],
            section_name="Pricing",
        )
        team_tasks.append(pricing_research_results)
        # get the recent events results
        recent_events_research_results = self.tasks.vector_multi_search_task(
            name="Recent Events Search",
            agent=vector_search_agent,
            search_query="What are the most recent events for this company?",
            instructions="Include any recent news, press releases, or other notable events including mergers and acquisitions.",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                personnel_research_results,
                competitors_research_results,
                company_history_research_results,
                pricing_research_results,
            ],
            section_name="Recent Events",
        )
        team_tasks.append(recent_events_research_results)
        # get the products and services results
        products_services_research_results = self.tasks.vector_multi_search_task(
            name="Products and Services Search",
            agent=vector_search_agent,
            search_query="What are the key products and services for this company?",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                personnel_research_results,
                competitors_research_results,
                company_history_research_results,
                pricing_research_results,
                recent_events_research_results,
            ],
            section_name="Products and Services",
        )
        team_tasks.append(products_services_research_results)
        # get the market results
        market_research_results = self.tasks.vector_multi_search_task(
            name="Market Search",
            agent=vector_search_agent,
            search_query="What market does this company operate in? What is their TAM/SAM/SOM?",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                personnel_research_results,
                competitors_research_results,
                company_history_research_results,
                pricing_research_results,
                recent_events_research_results,
                products_services_research_results,
            ],
            section_name="Market Analysis",
        )
        team_tasks.append(market_research_results)
        # get the swot results from vector search
        swot_research_results = self.tasks.vector_multi_search_task(
            name="SWOT Search",
            agent=vector_search_agent,
            search_query="What are strengths, weaknesses, opportunities, and threats for this company?",
            context=[
                company_determination_search_task,
                company_structure_research_results,
                personnel_research_results,
                competitors_research_results,
                company_history_research_results,
                pricing_research_results,
                recent_events_research_results,
                products_services_research_results,
                market_research_results,
            ],
            section_name="SWOT Analysis",
        )
        team_tasks.append(swot_research_results)
        # get the swot company information
        return TeamTaskAssignment(
            team=team,
            tasks=team_tasks,
        )

    def run(self) -> CrewRun:
        # create team tasking so i can access tasks for context later
        company_determination_team = self._build_determination_task()
        # get the company determination task
        determined_company_task = company_determination_team.tasks[-1]
        self.run_team_task(company_determination_team)
        # create a search team for the desired sections
        search_team = self.build_search_task(
            company_determination_search_task=determined_company_task
        )
        # self.run_search_task(search_team)
        for crew in search_team:
            crew.kickoff()
        # build research team
        print("Building research team ...")
        research_team = self.build_research_task(
            company_determination_search_task=determined_company_task
        )
        print("Kicking off research team ...")
        # run the search task
        if self.cache:
            crew = RedisCacheHandlerCrew(
                agents=research_team.team,
                tasks=research_team.tasks,
                cache=self.cache,
                _cache_handler=self.cache_handler,
                task_callback=self.task_callback,
            )
        else:
            crew = Crew(
                agents=research_team.team,
                tasks=research_team.tasks,
                task_callback=self.task_callback,
            )
        result = crew.kickoff()
        # create and return crew ru n
        crew_run = CrewRun(
            tasks=[task_to_task_run(task) for task in crew.tasks],
            result=result.raw,
            token_usage=result.token_usage,
        )
        return crew_run
