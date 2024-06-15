"""
Marketing agents
"""
from crewai import Agent
from crewai_tools.tools import ScrapeWebsiteTool
from conductor.crews.marketing.tools import SerpSearchTool


class MarketingAgents:
    """
    Marketing agents
    """

    def swot_agent(self, llm=None):
        return Agent(
            role="SWOT Agent",
            goal="Create a SWOT analysis for a company",
            backstory="An expert in analyzing company data to create SWOT analysis",
            verbose=True,
            llm=llm,
        )

    def search_engine_agent(self, llm=None):
        return Agent(
            role="Search Engine Agent",
            goal="Provide URLs for a additional company research and extract all useful research information to support company analysis",
            backstory="An expert in searching for information about companies. Generate great search engine questions that will get the best results. Provide summaries with key information about the company.",
            verbose=True,
            tools=[SerpSearchTool()],
            llm=llm,
        )

    def company_research_agent(self, llm=None):
        return Agent(
            role="Company Agent",
            goal="Retrieve information about a company",
            backstory="An expert in looking up company information using the internet.",
            verbose=True,
            tools=[ScrapeWebsiteTool(), SerpSearchTool()],
            llm=llm,
        )

    def competitor_agent(self, llm=None):
        return Agent(
            role="Competitor Agent",
            goal="Retrieve information about a companies competitor",
            backstory="An expert in looking up competitor information",
            tools=[ScrapeWebsiteTool(), SerpSearchTool()],
            verbose=True,
            llm=llm,
        )

    def writer_agent(self, llm=None):
        return Agent(
            role="Writer Agent",
            goal="Write a report about a company using the provided context in a task.",
            backstory="An expert in writing comprehensive reports about companies. Focuses on key points but also provides a detailed analysis. Also includes all sources used.",
            verbose=True,
            llm=llm,
            allow_delegation=False,
        )
