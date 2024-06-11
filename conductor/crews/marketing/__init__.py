from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.reports import outputs


def url_marketing_report(url: str) -> str:
    """
    Run a marketing report on a URL
    """
    crew = UrlMarketingCrew(url)
    result = crew.run()
    report = outputs.string_to_report(result)
    return report
