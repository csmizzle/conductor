from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.reports.outputs import string_to_report
from conductor.reports.models import Report, ReportStyle


def url_marketing_report(
    url: str,
    report_style: ReportStyle,
    haiku: bool = True,
) -> Report:
    """
    Run a marketing report on a URL
    """
    crew = UrlMarketingCrew(
        url=url,
        report_style=report_style,
    )
    result = crew.run()
    try:
        report = string_to_report(
            string=result,
            report_style=report_style,
            haiku=haiku,
        )
    except Exception as e:
        print("Error parsing report. Returning report with raw output.")
        print(e)
        report = Report(
            title="",
            description="",
            sections=[],
            raw=result,
            style=report_style,
        )
    return report
