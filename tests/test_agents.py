"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.crews.models import CrewRun, TaskRun
from conductor.reports.models import ReportStyle
from conductor.crews.cache import RedisCrewCacheHandler
import os


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


def test_url_marketing_crew_with_redis_cache():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(
        url=url,
        report_style=ReportStyle.BULLETED,
        cache=True,
        cache_handler=RedisCrewCacheHandler(url=os.getenv("REDIS_CREW_CACHE_URL")),
    )
    result = crew.run()
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)
