"""
Marketing agents
"""
from crewai import Agent
from crewai_tools.tools.base_tool import BaseTool
from pydantic import InstanceOf
from crewai_tools import ScrapeWebsiteTool
from conductor.crews.marketing.tools import (
    SerpSearchTool,
    SerpSearchCacheTool,
    ScrapePageCacheTool,
    SerpSearchOxyLabsTool,
    ScrapePageOxyLabsTool,
    SerpSearchOxylabsCacheTool,
    ScrapePageOxylabsCacheTool,
)


class MarketingAgents:
    """
    Marketing agents
    """

    @staticmethod
    def set_scraping_tools(*, cache: bool, proxy: bool) -> list[InstanceOf[BaseTool]]:
        """
        Set agent tools based on crew parameters
        """
        if cache and proxy:
            return [SerpSearchOxylabsCacheTool(), ScrapePageOxylabsCacheTool()]
        elif proxy and not cache:
            return [SerpSearchOxyLabsTool(), ScrapePageOxyLabsTool()]
        elif cache and not proxy:
            return [SerpSearchCacheTool(), ScrapePageCacheTool()]
        else:
            return [SerpSearchTool(), ScrapeWebsiteTool()]

    def swot_agent(self, llm=None, cache=None, proxy=None, cache_handler=None) -> Agent:
        return Agent(
            role="SWOT Agent",
            goal="Create a SWOT analysis for a company and returning source links.",
            backstory="An expert in analyzing company data to create SWOT analysis.",
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            verbose=True,
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def search_engine_agent(self, llm=None, cache=None, proxy=None, cache_handler=None):
        return Agent(
            role="Search Engine Agent",
            goal="Provide URLs for a additional company research and extract all useful research information to support company analysis. Search engine queries should be analytical and insightful.",
            backstory="An expert in searching for information about companies. Generate great search engine questions that will get the best results. Provide summaries with key information about the company.",
            verbose=True,
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def company_research_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ):
        return Agent(
            role="Company Agent",
            goal="Retrieve comprehensive information about a company and returning source links.",
            backstory="An expert in looking up company information using the internet. Comes up with creative ways to find information about a company.",
            verbose=True,
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def competitor_agent(self, llm=None, cache=None, proxy=None, cache_handler=None):
        return Agent(
            role="Competitor Agent",
            goal="Retrieve information about a companies competitor",
            backstory="An expert in looking up competitor information",
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            verbose=True,
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def writer_agent(self, llm=None, cache=None, cache_handler=None):
        return Agent(
            role="Writer Agent",
            goal="Write a report about a company using the provided context in a task.",
            backstory="An expert in writing comprehensive reports about companies. Focuses on key points but also provides a detailed analysis. Also includes all sources used.",
            verbose=True,
            llm=llm,
            allow_delegation=False,
            cache=cache,
            cache_handler=cache_handler,
        )
