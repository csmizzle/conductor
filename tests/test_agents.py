"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew
from tests.constants import BASEDIR
from conductor.crews.models import CrewRun, TaskRun
from conductor.reports.models import ReportStyle
import vcr
import os


@vcr.use_cassette(os.path.join(BASEDIR, "cassettes", "test_url_marketing_crew.yaml"))
def test_url_marketing_crew():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(url=url, report_style=ReportStyle.BULLETED)
    result = crew.run()
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)
