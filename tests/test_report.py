from conductor.reports.generators import ConductorUrlReportGenerator, Paragraph
from test_vars import TEST_REPORT_RESPONSE
from conductor.reports.html_ import report_to_html
from conductor.reports.outputs import (
    report_to_pdf,
    report_to_pdf_binary,
    string_to_report,
)
import os


def test_conductor_url_report_generator():
    urls = ["trssllc.com"]
    report_title = "Test Report"
    report_description = "This is a test report."
    generator = ConductorUrlReportGenerator(
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
        report_title=report_title,
        report_description=report_description,
    )
    report = generator.generate()
    assert report.title == report_title
    assert report.description == report_description
    assert len(report.paragraphs) == 2
    assert report.paragraphs[0].title == f"Website Summary of {urls[0]}"
    assert report.paragraphs[1].title == f"Personnel Summary of {urls[0]}"
    assert report.paragraphs[0].content is not None
    assert report.paragraphs[1].content is not None


def test_report_to_html() -> None:
    urls = ["trssllc.com"]
    report_title = "Test Report"
    report_description = "This is a test report."
    generator = ConductorUrlReportGenerator(
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
        report_title=report_title,
        report_description=report_description,
    )
    report = generator.generate()
    html = report_to_html(report)
    assert report_title in html
    assert report_description in html
    assert report.paragraphs[0].title in html
    assert report.paragraphs[0].content in html
    assert report.paragraphs[1].title in html
    assert report.paragraphs[1].content in html


def test_report_to_pdf() -> None:
    urls = ["trssllc.com"]
    report_title = "Test Report"
    report_description = "This is a test report."
    generator = ConductorUrlReportGenerator(
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
        report_title=report_title,
        report_description=report_description,
    )
    report = generator.generate()
    report_to_pdf(report, "test.pdf")
    os.remove("test.pdf")
    assert not os.path.exists("test.pdf")


def test_report_to_pdf_binary() -> None:
    urls = ["trssllc.com"]
    report_title = "Test Report"
    report_description = "This is a test report."
    generator = ConductorUrlReportGenerator(
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
        report_title=report_title,
        report_description=report_description,
    )
    report = generator.generate()
    binary_ = report_to_pdf_binary(report)
    return binary_


def test_string_to_response() -> None:
    report = string_to_report(TEST_REPORT_RESPONSE)
    print(report)


test_string_to_response()
