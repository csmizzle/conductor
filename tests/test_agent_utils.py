from conductor.crews.marketing.utils import create_report_prompt
from conductor.reports.models import ReportStyle


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
