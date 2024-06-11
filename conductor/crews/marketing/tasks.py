"""
Tasks for the marketing crew.
"""
from crewai import Task, Agent
from textwrap import dedent


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
            expected_output="Additional URLs and data for more company research.",
        )

    def company_research_task(self, agent: Agent, company_url: str):
        return Task(
            description=dedent(
                f"""
            Determine which company the URL belongs to and do in-depth research on the company.
            Find key personnel, company history, any pricing information available, and key products and services.
            Also find any competitors as well.
            Use the URL: {company_url} to find the company.
            """
            ),
            agent=agent,
            expected_output="Detailed company information.",
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
            expected_output="SWOT analysis of the company.",
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
            expected_output="List of competitors with strengths and weaknesses.",
        )

    def company_report_task(self, agent: Agent, context: list[Task]):
        return Task(
            description=dedent(
                """
            Write a comprehensive report on the company.
            The report should be a mixture of long form and bullet points, capturing the key points.
            The report should include a SWOT analysis, key personnel, company history, and key products and services.
            The report should be well-structured and easy to read.
            The report should be broken in three main sections:
                - Overview
                    - Background
                    - Key Personnel
                    - Products/Services
                    - Pricing
                - SWOT Analysis
                    - Strengths
                    - Weaknesses
                    - Opportunities
                    - Threats
                - Competitors
                    - Competitor
                        - Strengths
                        - Weaknesses
            """
            ),
            agent=agent,
            context=context,
            expected_output="Comprehensive report on the company with the sections overview, swot analysis, and competitors.",
        )
