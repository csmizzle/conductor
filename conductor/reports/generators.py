"""
Use Conductor tools, chains, and prompts to generate reports
"""
from conductor.reports.models import Report, Paragraph, Generator


class ConductorUrlReportGenerator(Generator):
    """
    Take a number or urls and generate a report with the following data:
    - Scraped data and summarization
    -
    """

    def __init__(
        self,
        report_title: str,
        report_description: str,
        paragraphs: list[Paragraph] = [],
    ) -> None:
        self.report_title = report_title
        self.report_description = report_description
        self.paragraphs = paragraphs

    def generate(self) -> Report:
        """
        Generate a simple report with Apollo Data
        """
        return Report(
            title=self.report_title,
            description=self.report_description,
            paragraphs=self.paragraphs,
        )
