from conductor.crews.marketing import url_marketing_report
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    report = url_marketing_report("https://www.trssllc.com")
    with open(os.path.join(BASEDIR, "data", "output.txt"), "w") as f:
        f.write(report.raw)
