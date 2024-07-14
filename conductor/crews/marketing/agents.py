"""
Marketing agents
"""
from crewai import Agent
from crewai_tools.tools.base_tool import BaseTool
from pydantic import InstanceOf
from crewai_tools import ScrapeWebsiteTool
from conductor.crews.marketing.tools import (
    SerpSearchTool,
    SerpBingSearchTool,
    SerpSearchCacheTool,
    ScrapePageCacheTool,
    SerpBingSearchCacheTool,
    SerpSearchOxyLabsTool,
    ScrapePageOxyLabsTool,
    SerpBingSearchOxyLabsTool,
    SerpSearchOxylabsCacheTool,
    ScrapePageOxylabsCacheTool,
    SerpBingSearchOxylabsCacheTool,
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
            return [
                SerpSearchOxylabsCacheTool(),
                ScrapePageOxylabsCacheTool(),
                SerpBingSearchOxylabsCacheTool(),
            ]
        elif proxy and not cache:
            return [
                SerpSearchOxyLabsTool(),
                ScrapePageOxyLabsTool(),
                SerpBingSearchOxyLabsTool(),
            ]
        elif cache and not proxy:
            return [
                SerpSearchCacheTool(),
                ScrapePageCacheTool(),
                SerpBingSearchCacheTool(),
            ]
        else:
            return [SerpSearchTool(), ScrapeWebsiteTool(), SerpBingSearchTool()]

    def url_research_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ) -> Agent:
        return Agent(
            role="URL Research Agent",
            goal="Determine what company a URL belongs to.",
            backstory="An expert in finding out what company a URL belongs to using multiple search engines. Uses the URL to find the company and then provides detailed information about the company.",
            tools=[
                self.set_scraping_tools(cache=cache, proxy=proxy)[1]
            ],  # get only the ScrapePage tool
            verbose=True,
            cache=cache,
            cache_handler=cache_handler,
            llm=llm,
        )

    def swot_agent(self, llm=None, cache=None, proxy=None, cache_handler=None) -> Agent:
        return Agent(
            role="SWOT Agent",
            goal="Create a SWOT analysis for a company and returning source links.",
            backstory="An expert in analyzing company data to create SWOT analysis using all available context and tools.",
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            verbose=True,
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def search_engine_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ) -> Agent:
        return Agent(
            role="Search Engine Agent",
            goal="Provide URLs for a additional company research and extract all useful research information to support company analysis. Search engine queries should be analytical and insightful.",
            backstory="An expert in searching for information about companies. Generate great search engine questions across multiple search engines that will get the best results. Provide summaries with key information about the company.",
            verbose=True,
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def company_research_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ) -> Agent:
        return Agent(
            role="Company Agent",
            goal="Retrieve comprehensive information about a company and returning source links.",
            backstory="An expert in looking up company information using the internet. Comes up with creative ways to find information about a company across multiple search engines. Uses advanced research techniques to find the most relevant information.",
            verbose=True,
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def competitor_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ) -> Agent:
        return Agent(
            role="Competitor Agent",
            goal="Retrieve information about a companies competitor",
            backstory="An expert in looking up competitor information that is relevant to the company. Uses advanced research techniques to find the most relevant information. Finds companies that are direct competitors to the company based on relevant criteria.",
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            verbose=True,
            llm=llm,
            cache=cache,
            cache_handler=cache_handler,
        )

    def writer_agent(self, llm=None, cache=None, cache_handler=None) -> Agent:
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

    def key_questions_answerer_agent(
        self, llm=None, cache=None, proxy=None, cache_handler=None
    ) -> Agent:
        return Agent(
            role="Key Questions Agent",
            goal="Answer key questions about a company using the provided context in a task.",
            backstory="An expert in answering key questions about companies. Focuses on key points but also provides a detailed analysis. Addresses common analytical pitfalls during its approach. Also includes all sources used and looks across multiple search engines.",
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=self.set_scraping_tools(cache=cache, proxy=proxy),
            cache=cache,
            cache_handler=cache_handler,
        )

    def editor_agent(self, llm=None) -> Agent:
        return Agent(
            role="Editor Agent",
            goal="Edit a report about a company. Make sure the statements are accurate, sourced, and well-written.",
            backstory="An expert in editing reports. Makes sure the report is accurate, well-written, and sourced. Use the provided context and tools to make the report better and more accurate.",
            verbose=True,
            llm=llm,
            allow_delegation=True,
        )
