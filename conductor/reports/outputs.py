"""
Conductor Report Outputs
- PDF Report
"""
import pdfkit
from conductor.reports.models import Report
from conductor.reports.html_ import report_to_html
import tempfile


def report_to_pdf(report: Report, filename: str) -> bytes:
    """
    Convert a report to a PDF
    """
    html = report_to_html(report)
    pdf = pdfkit.from_string(html, filename)
    return pdf


def report_to_pdf_binary(report: Report) -> bytes:
    """
    Convert a report to a PDF
    """
    html = report_to_html(report)
    with tempfile.NamedTemporaryFile() as f:
        pdfkit.from_string(html, f.name)
        return f.read()
