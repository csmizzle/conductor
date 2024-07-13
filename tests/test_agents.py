"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.crews.marketing import run_marketing_crew
from conductor.crews.marketing.agents import MarketingAgents
from conductor.crews.marketing.tools import (
    SerpSearchOxylabsCacheTool,
    ScrapePageOxylabsCacheTool,
    SerpSearchOxyLabsTool,
    ScrapePageOxyLabsTool,
    SerpSearchCacheTool,
    ScrapePageCacheTool,
    SerpSearchTool,
)
from crewai_tools import ScrapeWebsiteTool
from conductor.crews.models import CrewRun, TaskRun
from conductor.reports.models import ReportStyle


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


def test_url_with_key_questions_marketing_crew():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    key_questions = [
        "Who could potentially purchase this company?",
        "Which industry does the company operate in?",
    ]
    crew = UrlMarketingCrew(
        url=url,
        report_style=ReportStyle.BULLETED,
        key_questions=key_questions,
    )
    result = crew.run()
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)


def test_url_marketing_crew_with_proxy():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(
        url=url,
        report_style=ReportStyle.BULLETED,
        proxy=True,
    )
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
        redis=True,
    )
    result = crew.run()
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)


def test_url_marketing_crew_with_redis_and_proxy():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(
        url=url,
        report_style=ReportStyle.BULLETED,
        cache=True,
        proxy=True,
        redis=True,
    )
    result = crew.run()
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)


def test_set_scraping_tools() -> None:
    """
    Test that the set_scraping_tools function returns the correct tools.
    """
    # both proxy and cache
    tools = MarketingAgents.set_scraping_tools(cache=True, proxy=True)
    assert len(tools) == 2
    assert isinstance(tools[0], SerpSearchOxylabsCacheTool)
    assert isinstance(tools[1], ScrapePageOxylabsCacheTool)
    # proxy only
    tools = MarketingAgents.set_scraping_tools(cache=False, proxy=True)
    assert len(tools) == 2
    assert isinstance(tools[0], SerpSearchOxyLabsTool)
    assert isinstance(tools[1], ScrapePageOxyLabsTool)
    # cache only
    tools = MarketingAgents.set_scraping_tools(cache=True, proxy=False)
    assert len(tools) == 2
    assert isinstance(tools[0], SerpSearchCacheTool)
    assert isinstance(tools[1], ScrapePageCacheTool)
    # no proxy or cache
    tools = MarketingAgents.set_scraping_tools(cache=False, proxy=False)
    assert len(tools) == 2
    assert isinstance(tools[0], SerpSearchTool)
    assert isinstance(tools[1], ScrapeWebsiteTool)


def test_run_marketing_crew_with_proxy_and_cache() -> None:
    """
    Test the run_marketing_crew function with proxy and cache.
    """
    url = "https://www.trssllc.com"
    result = run_marketing_crew(
        url=url,
        report_style=ReportStyle.BULLETED,
        cache=True,
        proxy=True,
    )
    assert isinstance(result, CrewRun)
    assert result.result is not None
    assert isinstance(result.result, str)
    for task in result.tasks:
        assert isinstance(task, TaskRun)
