from conductor.reports.generators import ConductorUrlReportGenerator, Paragraph


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
