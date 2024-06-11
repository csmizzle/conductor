"""
Test the agents module.
"""
from conductor.crews.marketing.crew import UrlMarketingCrew


def test_url_marketing_crew():
    """
    Test the UrlMarketingCrew class.
    """
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(url)
    result = crew.run()
    assert isinstance(result, str)
    return result


result = test_url_marketing_crew()
with open("./output.txt", "w") as f:
    f.write(result)
