"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.crews.marketing import url_marketing_report
from conductor.reports.models import Report, ParsedReport, ReportStyle


def test_url_marketing_crew():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(url)
    result = crew.run()
    assert isinstance(result, str)
    return result


def test_url_marketing_report():
    """
    Test the url_marketing_report function.
    """
    url = "https://www.bytedance.com/en/"
    report = url_marketing_report(url, ReportStyle.BULLETED)
    assert isinstance(report, Report)
    assert isinstance(report.report, ParsedReport)
    assert report.style == "BULLETED"
    assert isinstance(report.raw, str)


test_url_marketing_report()
