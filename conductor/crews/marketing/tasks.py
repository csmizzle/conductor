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

    def company_identification_task(self, agent: Agent, company_url: str) -> Task:
        return Task(
            description=dedent(
                f"""
            Determine what company a URL belongs to.
            Use the URL to find the company and then provide detailed information about the company.
            URL to find the company: {company_url}
            """
            ),
            agent=agent,
            expected_output="Detailed company identification with source links.",
        )

    def search_engine_task(self, agent: Agent, context: list[Task]) -> Task:
        return Task(
            description=dedent(
                """
            Find additional links that can be used to research the company.
            Ask questions that will lead to more information about the company, industry, and competitors.
            These links should be passed to the company research agent for further investigation.
            """
            ),
            agent=agent,
            context=context,
            expected_output="Additional URLs and insights for more company research with source links.",
        )

    def company_research_task(
        self, agent: Agent, company_url: str, context: list[Task]
    ) -> Task:
        return Task(
            description=dedent(
                f"""
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
            context=context,
        )

    def answer_key_questions_task(
        self, agent: Agent, key_questions: str, context: list[Task]
    ) -> Task:
        return Task(
            description=dedent(
                f"""
            Answer the provided key questions about the company.
            To find the answers, use creative but detailed research techniques using the provided tools.
            The answers should be detailed and provide a comprehensive overview of the company.
            {key_questions}
            """
            ),
            agent=agent,
            context=context,
            expected_output="Answers to key questions about the company with source links.",
        )

    def company_swot_task(self, agent: Agent, context: list[Task]) -> Task:
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

    def company_competitor_task(self, agent: Agent, context: list[Task]) -> Task:
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
    ) -> Task:
        return Task(
            description=create_report_prompt(report_style),
            agent=agent,
            context=context,
            expected_output="Comprehensive report on the company with the sections overview, swot analysis, and competitors.",
        )

    def review_task(
        self,
        company_url: str,
        agent: Agent,
        context: list[Task],
        report_style: ReportStyle,
    ) -> Task:
        return Task(
            description=dedent(
                f"""
            Review and make changes to final report ensuring three main concepts:
            - The report is well-structured and easy to read.
                - The edits of the report should not change the structure of the report.
            - The report includes all key points and important details.
            - The report includes accurate, up to date, and relevant information for this URL {company_url}.
                - If similar companies are found, make sure to differentiate the information but focus on {company_url}.
            Ensure the final report stays true to this style and structure:
            {create_report_prompt(report_style)}
            """
            ),
            agent=agent,
            context=context,
            expected_output="Final reviewed report ready for delivery.",
        )
