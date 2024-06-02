"""
HTML Tooling for reports
"""
from conductor.reports.models import Report
import html


def report_to_html(report: Report) -> str:
    """
    Convert a report to an HTML string
    """
    html_string = f"""
    <html>
        <head>
            <title>{report.title}</title>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p>{report.description}</p>
            <ul>
    """
    for paragraph in report.paragraphs:
        html_string += f"<li><h2>{paragraph.title}</h2><p>{paragraph.content}</p></li>"
    html_string += """
            </ul>
        </body>
    </html>
    """
    return html.unescape(html_string)
