from tests.constants import TEST_REPORT_RESPONSE
from conductor.reports.outputs import string_to_report
import json
import os


BASEDIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    report = string_to_report(TEST_REPORT_RESPONSE)
    with open(os.path.join(BASEDIR, "data", "json_report.json"), "w") as f:
        f.write(json.dumps(report.dict(), indent=4))
