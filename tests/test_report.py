from tests.constants import TEST_REPORT_RESPONSE, REPORT_JSON, TEST_COMPLEX_NARRATIVE
from conductor.reports.outputs import (
    string_to_report,
)
from conductor.reports.models import ParsedReport, Report, Section, Paragraph
from conductor.reports.html_ import report_to_html
from langsmith import unit


def test_conductor_url_report_generator():
    urls = ["trssllc.com"]
    report_title = "Test Report"
    report_description = "This is a test report."
    section_title = "This is a test section title"
    report = Report(
        title=report_title,
        description=report_description,
        sections=[
            Section(
                title=section_title,
                paragraphs=[
                    Paragraph(
                        title=f"Website Summary of {urls[0]}",
                        content="This is a summary of the website.",
                    ),
                    Paragraph(
                        title=f"Personnel Summary of {urls[0]}",
                        content="This is a summary of the personnel.",
                    ),
                ],
            )
        ],
    )
    assert report.title == report_title
    assert report.description == report_description
    assert len(report.sections) == 1
    assert report.sections[0].title == section_title
    assert len(report.sections[0].paragraphs) == 2
    assert report.sections[0].paragraphs[0].title == f"Website Summary of {urls[0]}"
    assert report.sections[0].paragraphs[1].title == f"Personnel Summary of {urls[0]}"
    assert report.sections[0].paragraphs[0].content is not None
    assert report.sections[0].paragraphs[1].content is not None


@unit
def test_string_to_report() -> None:
    report = string_to_report(
        TEST_REPORT_RESPONSE,
    )
    assert isinstance(report, ParsedReport)
    # assert report.title == "Thomson Reuters Special Services (TRSS) Company Report"
    # # test for section structure
    # assert len(report.sections) == 5
    # # Overview Section
    # assert report.sections[0].title == "1. Overview"
    # assert len(report.sections[0].paragraphs) == 4
    # # Market Analysis Section
    # assert report.sections[1].title == "2. Market Analysis"
    # assert len(report.sections[1].paragraphs) == 2
    # # SWOT Analysis Section
    # assert report.sections[2].title == "3. SWOT Analysis"
    # assert len(report.sections[2].paragraphs) == 4
    # # Competitors Section
    # assert report.sections[3].title == "4. Competitors"
    # assert len(report.sections[3].paragraphs) == 2
    # assert report.raw == TEST_REPORT_RESPONSE
    # # Sources Section
    # assert report.sections[4].title == "5. Sources"
    # assert len(report.sections[4].paragraphs) == 1


def test_report_to_html() -> None:
    # read in the json data
    parsed_report = ParsedReport(**REPORT_JSON)
    report = Report(report=parsed_report)
    html = report_to_html(report)
    assert isinstance(html, str)


def test_complex_narrative_to_report() -> None:
    report = string_to_report(
        TEST_COMPLEX_NARRATIVE,
    )
    assert isinstance(report, ParsedReport)
