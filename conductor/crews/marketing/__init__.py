from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.reports import outputs
from conductor.reports.models import Report, ReportStyle


def url_marketing_report(
    url: str, report_style: ReportStyle = ReportStyle.BULLETED
) -> Report:
    """
    Run a marketing report on a URL
    """
    crew = UrlMarketingCrew(url=url, report_style=report_style)
    result = crew.run()
    report = outputs.string_to_report(result)
    return report
