"""
Tasks for the marketing crew.
"""
from crewai import Task, Agent
from textwrap import dedent
from conductor.reports.models import ReportStyle
from conductor.crews.marketing.utils import create_report_prompt


class MarketingTasks:
    """
    Marketing crew
    """

    def search_engine_task(self, agent: Agent, context: list[Task]):
        return Task(
            description=dedent(
                """
            Find additional links that can be used to research the company.
            These links should be passed to the company research agent for further investigation.
            """
            ),
            agent=agent,
            context=context,
            expected_output="Additional URLs and data for more company research with source links.",
        )

    def company_research_task(self, agent: Agent, company_url: str):
        return Task(
            description=dedent(
                f"""
            Determine which company the URL belongs to and do in-depth research on the company.
            Find key personnel, company history, any pricing information available, and key products and services.
            Look for the key personnel on the linkedin or company website.
            Find any relevant contact information with key personnel.
            Find the competitors of the company.
            Find the most recent events of the company as well.
            Determine which market the company operates in and what their TAM/SAM/SOM is.
            Find estimates if exact numbers are not available.
            Use the URL: {company_url} to find the company.
            """
            ),
            agent=agent,
            expected_output="Detailed company information with source links.",
        )

    def company_swot_task(self, agent: Agent, context: list[Task]):
        return Task(
            description=dedent(
                """
            Break down the company's strengths, weaknesses, opportunities, and threats.
            Each section should be detailed and provide a comprehensive analysis.
            The analysis should always be broken into four main sections.
            """
            ),
            agent=agent,
            context=context,
            expected_output="SWOT analysis of the company with source links.",
        )

    def company_competitor_task(self, agent: Agent, context: list[Task]):
        return Task(
            description=dedent(
                """
            Find the top competitors for the company in the provided context.
            Provide a brief overview of each competitor, including their strengths and weaknesses.
            """
            ),
            agent=agent,
            context=context,
            expected_output="List of competitors with strengths and weaknesses and source links.",
        )

    def company_report_task(
        self, agent: Agent, context: list[Task], report_style: ReportStyle
    ):
        return Task(
            description=create_report_prompt(report_style),
            agent=agent,
            context=context,
            expected_output="Comprehensive report on the company with the sections overview, swot analysis, and competitors.",
        )
