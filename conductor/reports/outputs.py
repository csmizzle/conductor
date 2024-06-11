"""
Conductor Report Outputs
- PDF Report
"""
import pdfkit
from langchain.prompts import PromptTemplate
from conductor.reports.models import Report
from conductor.reports.html_ import report_to_html
from conductor.llms import claude_sonnet
from textwrap import dedent
import tempfile
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.llm import LLMChain


def string_to_report(string: str) -> Report:
    report_parser = PydanticOutputParser(pydantic_object=Report)
    string_to_report_prompt = PromptTemplate(
        input_variables=["string"],
        template=dedent(
            """
        Convert the following string to a Report JSON object:
        {string}
        \n
        {format_instructions}
        """
        ),
        partial_variables={
            "format_instructions": report_parser.get_format_instructions()
        },
    )
    chain = LLMChain(
        llm=claude_sonnet,
        prompt=string_to_report_prompt,
    )
    response = chain.invoke({"string": string})
    return report_parser.parse(text=response["text"])


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
