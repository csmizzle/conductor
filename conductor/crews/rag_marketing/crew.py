from crewai import Crew, Agent, Task
from conductor.crews.rag_marketing.agents import MarketingRagAgents
from conductor.crews.rag_marketing.tasks import RagMarketingTasks
from conductor.crews.marketing.utils import task_to_task_run
from conductor.crews.models import CrewRun
from conductor.crews.cache import RedisCrewCacheHandler
from conductor.crews.handlers import RedisCacheHandlerCrew
from crewai.agents.cache.cache_handler import CacheHandler
from conductor.llms import claude_sonnet
from elasticsearch import Elasticsearch
from typing import Callable


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
        self.elasticsearch = elasticsearch
        self.index_name = index_name
        self.cache = cache
        if redis:
            self.cache_handler = RedisCrewCacheHandler()
        else:
            self.cache_handler = CacheHandler()
        self.task_callback = task_callback

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
        vector_meta_search_agent = agents.vector_search_metadata_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        # get company structure, personnel, swot, competitors, company history, pricing, recent events, products, services, market
        company_structure_research_agent = agents.company_structure_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        personnel_research_agent = agents.personnel_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        swot_research_agent = agents.swot_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        competitors_research_agent = agents.competitor_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        company_history_research_agent = agents.company_history_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        pricing_research_agent = agents.pricing_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        recent_events_research_agent = agents.recent_events_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        products_services_research_agent = agents.product_and_services_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )
        market_research_agent = agents.market_research_agent(
            elasticsearch=self.elasticsearch,
            index_name=self.index_name,
            llm=claude_sonnet,
        )

        team.append(data_collection_agent)
        team.append(swot_research_agent)
        team.append(vector_search_agent)
        team.append(company_structure_research_agent)
        team.append(personnel_research_agent)
        team.append(competitors_research_agent)
        team.append(company_history_research_agent)
        team.append(pricing_research_agent)
        team.append(recent_events_research_agent)
        team.append(products_services_research_agent)
        team.append(market_research_agent)
        # create all tasks and add them to the team
        team_tasks = []
        url_collection_task = tasks.url_collection_task(
            name="URL Collection Research",
            agent=data_collection_agent,
            company_url=self.url,
        )
        team_tasks.append(url_collection_task)
        # determine who the company is
        company_determination_search_task = tasks.vector_metadata_search_task(
            name="Company Determination Research",
            agent=vector_meta_search_agent,
            url=self.url,
        )
        team_tasks.append(company_determination_search_task)
        # get the company structure
        company_structure_research_task = tasks.company_structure_research_task(
            name="Company Structure Research",
            agent=company_structure_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(company_structure_research_task)
        # get the personnel information
        personnel_research_task = tasks.personnel_research_task(
            name="Personnel Research",
            agent=personnel_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(personnel_research_task)
        # get the competitors information
        competitors_research_task = tasks.competitor_research_task(
            name="Competitor Research",
            agent=competitors_research_agent,
            context=[
                company_determination_search_task,
                personnel_research_task,
                company_structure_research_task,
            ],
        )
        team_tasks.append(competitors_research_task)
        # get the company history information
        company_history_research_task = tasks.company_history_research_task(
            name="Company History Research",
            agent=company_history_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(company_history_research_task)
        # get the pricing information
        pricing_research_task = tasks.pricing_research_task(
            name="Pricing Research",
            agent=pricing_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(pricing_research_task)
        # get the recent events information
        recent_events_research_task = tasks.recent_events_research_task(
            name="Recent Events Research",
            agent=recent_events_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(recent_events_research_task)
        # get the products and services information
        products_services_research_task = tasks.products_and_services_research_task(
            name="Products and Service Research",
            agent=products_services_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(products_services_research_task)
        # market research information
        market_research_task = tasks.market_analysis_research_task(
            name="Market Research",
            agent=market_research_agent,
            context=[company_determination_search_task],
        )
        team_tasks.append(market_research_task)
        # get the company swot information
        swot_research_task = tasks.swot_research_task(
            name="SWOT Research",
            agent=swot_research_agent,
            context=[
                company_determination_search_task,
            ],
        )
        team_tasks.append(swot_research_task)
        # Question tasks
        # get the company structure results
        company_structure_research_results = tasks.vector_multi_search_task(
            name="Company Structure Search",
            agent=vector_search_agent,
            search_query="What is the structure of this company? Include things like beneficial owners, subsidiaries, and parent companies.",
            context=[company_determination_search_task],
            section_name="Company Structure",
        )
        team_tasks.append(company_structure_research_results)
        # get the company history results
        company_history_research_results = tasks.vector_multi_search_task(
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
        personnel_research_results = tasks.vector_multi_search_task(
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
        competitors_research_results = tasks.vector_multi_search_task(
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
        pricing_research_results = tasks.vector_multi_search_task(
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
        recent_events_research_results = tasks.vector_multi_search_task(
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
        products_services_research_results = tasks.vector_multi_search_task(
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
        market_research_results = tasks.vector_multi_search_task(
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
        swot_research_results = tasks.vector_multi_search_task(
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
        return team, team_tasks

    def run(self) -> CrewRun:
        team, team_tasks = self.build_team()
        if self.cache:
            crew = RedisCacheHandlerCrew(
                agents=team,
                tasks=team_tasks,
                cache=self.cache,
                _cache_handler=self.cache_handler,
                task_callback=self.task_callback,
            )
        else:
            crew = Crew(
                agents=team,
                tasks=team_tasks,
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
