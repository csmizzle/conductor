from constants import REPORT_JSON, BASEDIR
from conductor.reports.html_ import report_to_html
from conductor.reports.models import Report
import os


if __name__ == "__main__":
    report = Report(**REPORT_JSON)
    html = report_to_html(report)
    with open(os.path.join(BASEDIR, "data", "test_report.html"), "w") as f:
        f.write(html)
