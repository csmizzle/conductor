from conductor.reports.models import ReportStyle, ReportTemplatePromptGenerator
from textwrap import dedent


def create_report_prompt(
    report_style: ReportStyle, report_template_generator: ReportTemplatePromptGenerator
) -> str:
    return dedent(
        f"""
        Write a comprehensive report on the company using only the provided context.
        The report should be a world class and capture the key points along with important details.
        The report should be well-structured and easy to read.
        The report should include all source URLs used as well from the provided context, do not leave any out from provided context.
        The report should be broken into {len(report_template_generator.sections)} main sections and each section should be written {report_style.value}:
        {report_template_generator.generate()}
        """
    )
