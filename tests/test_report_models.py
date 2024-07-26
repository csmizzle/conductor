from conductor.reports.models import (
    ReportV2,
    SectionV2,
    ParagraphV2,
    ReportStyle,
    ReportTone,
    ParsedReportV2,
)


def test_report_v2() -> None:
    """
    Test and serialize a report
    """
    report = ReportV2(
        report=ParsedReportV2(
            title="Test Report",
            description="This is a test report.",
            sections=[
                SectionV2(
                    title="Test Section",
                    paragraphs=[
                        ParagraphV2(
                            title="Test Paragraph",
                            sentences=["This is a test sentence."],
                        )
                    ],
                    sources=["https://www.test.com"],
                    tone=ReportTone.PROFESSIONAL,
                    style=ReportStyle.NARRATIVE,
                )
            ],
        ),
        raw=["This is a raw report."],
    )
    report_dict = report.dict()
    assert isinstance(report_dict, dict)
