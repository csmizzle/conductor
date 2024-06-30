from conductor.crews.marketing.utils import (
    create_report_prompt,
    oxylabs_request,
    send_request,
)
from conductor.reports.models import ReportStyle
import os


def test_create_report_prompt() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    result = create_report_prompt(report_style=report_style)
    assert isinstance(result, str)
    assert (
        "as long form narratives, avoiding bullet points and short sentences." in result
    )
    return result


def test_oxylabs_request() -> None:
    oxylabs_username = os.getenv("OXYLABS_USERNAME")
    oxylabs_password = os.getenv("OXYLABS_PASSWORD")
    oxylabs_port = 7777
    oxylabs_response = oxylabs_request(
        method="GET",
        oxylabs_username=oxylabs_username,
        oxylabs_password=oxylabs_password,
        oxylabs_country="pr",
        oxylabs_port=oxylabs_port,
        url="https://ip.oxylabs.io/",
    )
    assert oxylabs_response.status_code == 200


def test_send_request() -> None:
    oxylabs_username = os.getenv("OXYLABS_USERNAME")
    oxylabs_password = os.getenv("OXYLABS_PASSWORD")
    response = send_request(
        method="GET",
        url="https://www.linkedin.com/company/trss",
        oxylabs_username=oxylabs_username,
        oxylabs_password=oxylabs_password,
    )
    assert response.status_code == 200
