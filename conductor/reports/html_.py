"""
HTML Tooling for reports
"""
from conductor.reports.models import Report
from jinja2 import Environment, FileSystemLoader
import html
import os


BASEDIR = os.path.dirname(os.path.abspath(__file__))


def report_to_html(report: Report) -> str:
    """
    Convert a report to an HTML string
    """
    env = Environment(loader=FileSystemLoader(os.path.join(BASEDIR, "templates")))
    report_template = env.get_template("report.html")
    output_from_parsed_template = report_template.render(
        report_title=report.title,
        report_description=report.description,
        report_sections=[section.dict() for section in report.sections],
    )
    return html.unescape(output_from_parsed_template)
