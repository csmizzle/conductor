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
    # write section and paragraphs to html
    for section in report.sections:
        html_string += f"<h2>{section.title}</h2>"
        for paragraph in section.paragraphs:
            html_string += f"<li><h3>{paragraph.title}</h3>"
            split_content = paragraph.content.split("\n")
            for content in split_content:
                html_string += f"\n<br><p>{content}</p></li>\n"
    # close html tags
    html_string += """
            </ul>
        </body>
    </html>
    """
    return html.unescape(html_string)
