"""
Marketing agents
"""
from crewai import Agent
from crewai_tools.tools import ScrapeWebsiteTool


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

    def company_research_agent(self, llm=None):
        return Agent(
            role="Company Agent",
            goal="Retrieve information about a company",
            backstory="An expert in looking up company information",
            verbose=True,
            tools=[ScrapeWebsiteTool()],
            llm=llm,
        )

    def competitor_agent(self, llm=None):
        return Agent(
            role="Competitor Agent",
            goal="Retrieve information about a companies competitor",
            backstory="An expert in looking up competitor information",
            verbose=True,
            llm=llm,
        )

    def writer_agent(self, llm=None):
        return Agent(
            role="Writer Agent",
            goal="Write a report about a company",
            backstory="An expert in writing comprehensive reports about companies",
            verbose=True,
            llm=llm,
        )
