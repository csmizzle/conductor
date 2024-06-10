"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew


def test_url_marketing_crew():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.google.com"
    crew = UrlMarketingCrew(url)
    result = crew.run()
    assert isinstance(result, str)


test_url_marketing_crew()
