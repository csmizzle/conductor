from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.reports.outputs import string_to_report
from conductor.reports.models import Report, ReportStyle


def url_marketing_report(
    url: str,
    report_style: ReportStyle,
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
        parsed_report = string_to_report(
            string=result,
        )
        report = Report(
            report=parsed_report,
            style=report_style,
            raw=result,
        )
    except Exception as e:
        print("Error parsing report. Returning report with raw output.")
        print(e)
        report = Report(
            report=None,
            style=report_style,
            raw=result,
        )
    return report
