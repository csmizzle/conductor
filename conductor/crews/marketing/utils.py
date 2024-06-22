from textwrap import dedent
from conductor.reports.models import ReportStyle
from conductor.crews.models import TaskRun
from crewai import Task


def create_report_prompt(report_style: ReportStyle):
    if report_style == ReportStyle.BULLETED:
        return dedent(
            f"""
        Write a comprehensive report on the company using only the provided context.
        The report should be a world class and capture the key points along with important details.
        The report should include a SWOT analysis, key personnel, company history, and key products and services.
        The report should be well-structured and easy to read.
        The report should include all source URLs used as well from the provided context, do not leave any out from provided context.
        The report should be broken into 5 main sections and each section should be written {report_style.value}:
            1. Overview
                - Background
                    - Background of the company in bullet points
                - Key Personnel
                    - All identified personnel in bullet points
                - Products/Services
                    - All identified products/services in bullet points
                - Pricing
                    - All identified pricing in bullet points
                - Recent Events
                    - All identified recent events in bullet points
            2. Market Analysis
                - Market
                    - Identified market(s) in bullet points
                - TAM/SAM/SOM
                    - All identified TAM/SAM/SOM in bullet points
            3. SWOT Analysis
                - Strengths
                    - All identified strengths in bullet points
                - Weaknesses
                    - All identified weaknesses in bullet points
                - Opportunities
                    - All identified opportunities in bullet points
                - Threats
                    - All identified threats in bullet points
            4. Competitors
                - Competitor
                    - All identified strengths, weaknesses, opportunities, and threats, for competitors in bullet points
            5. Sources
                - Links
        """
        )
    if report_style == ReportStyle.NARRATIVE:
        return dedent(
            f"""
            Write a comprehensive report on the company using only the provided context.
            The report should be a world class and capture the key points along with important details.
            The report should include a SWOT analysis, key personnel, company history, and key products and services.
            The report should be well-structured and easy to read.
            The report should include all source URLs used as well from the provided context, do not leave any out from provided context.
            The report should be broken into 5 main sections and each section should be written {report_style.value}:
                1. Overview
                    - Background
                        Paragraphs on the company background
                    - Key Personnel
                        Paragraphs on key personnel
                    - Products/Services
                        Paragraphs on products/services
                    - Pricing
                        Paragraphs on pricing
                    - Recent Events
                        Paragraphs on recent events
                2. Market Analysis
                    - Market
                        Identified market(s) in paragraphs
                    - TAM/SAM/SOM
                        Paragraphs on TAM/SAM/SOM
                3. SWOT Analysis
                    - Strengths
                        Paragraphs on strengths
                    - Weaknesses
                        Paragraphs on weaknesses
                    - Opportunities
                        Paragraphs on opportunities
                    - Threats
                        Paragraphs on threats
                4. Competitors
                    - Competitor
                        Paragraphs on competitors, strengths, weaknesses, opportunities, and threats
                5. Sources
                    - Links to sources as bullet points
            """
        )


def task_to_task_run(task: Task) -> TaskRun:
    return TaskRun(
        agent_role=task.agent.role,
        description=task.description,
        result=task.output.raw_output,
    )
