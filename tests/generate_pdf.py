from tests.constants import REPORT_JSON, BASEDIR
from conductor.reports.outputs import report_to_pdf
from conductor.reports.models import Report
import os


if __name__ == "__main__":
    report = Report(**REPORT_JSON)
    report_to_pdf(report, os.path.join(BASEDIR, "data", "test_report.pdf"))
