from conductor.reports.builder.outputs import PDFReportBuilder
from conductor.reports.builder import models
from tests.utils import load_model_from_test_data


def test_build_source_map() -> None:
    report = load_model_from_test_data(models.Report, "test_full_report_v3.json")
    builder = PDFReportBuilder(report, "test")
    source_map = builder.build_source_map()
    assert isinstance(source_map, dict)


def test_build_report() -> None:
    report = load_model_from_test_data(models.Report, "test_full_report_v3.json")
    builder = PDFReportBuilder(report, "Sentence Sourced Report")
    file_name = "test_report.pdf"
    builder.build(file_name)
