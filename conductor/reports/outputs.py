"""
Conductor Report Outputs
- PDF Report
"""
import pdfkit
from langchain.prompts import PromptTemplate
from conductor.reports.models import Report, ReportStyle
from conductor.reports.html_ import report_to_html
from conductor.llms import claude_sonnet, claude_haiku
from textwrap import dedent
import tempfile
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.llm import LLMChain
from langsmith import traceable


@traceable
def string_to_report(
    string: str, report_style: ReportStyle, haiku: bool = True
) -> Report:
    report_parser = PydanticOutputParser(pydantic_object=Report)
    string_to_report_prompt = PromptTemplate(
        input_variables=["string"],
        template=dedent(
            """
        Convert the following string to a Report JSON object.
        Do not modify the data, just parse it into the JSON object.
        Each section of the report should be separated by a newline character.
        Each Section should be be broken down into paragraphs.
        Include newlines in individual sections to that the sections maintain the original document structure when parsed into JSON.
        Always include all the sections and paragraphs from the original report.
        Always leave the raw field blank as this will be filled downstream.

        #### Example:
        Overview
        Background:
        Acme Corp is a company that specializes in making widgets.

        Acme was founded in 2008.
        Key Personnel:
        - John Doe is the CEO of Acme Corp.
        - Jim Dow is a CTO of Acme Corp.
        Pricing:
        Acme Corp's pricing is competitive.
        They also list their pricing publically.
        Products/Services:
        Acme Corp makes widgets.

        SWOT Analysis

        Strengths:
        1. Great product
        2. Strong Team

        Weaknesses:
        1. Bad location
        2. CEO low morale

        Opportunities:
        1. Cyber data
        2. Targeting new markets

        Threats:
        1. Strong competition
        2. Economic downturn

        #### Example JSON Object:
        {{
            "title": "Acme Corp Report",
            "description": "This is a report on Acme Corp.",
            "sections": [
                {{
                    "title": "Overview",
                    "paragraphs": [
                        {{
                            "title": "Background:"
                            "content": "\\nAcme Corp is a company that specializes in making widgets.\\nAcme was founded in 2008."
                        }},
                        {{
                            "title": "Key Personnel:",
                            "content": "\\n- John Doe is the CEO of Acme Corp.\\n- Jim Dow is a CTO of Acme Corp."
                        }},
                        {{
                            "title": "Pricing:",
                            "content": "\\nAcme Corp's pricing is competitive.\\nThey also list their pricing publically."
                        }}
                    ]
                }},
                {{
                    "title": "SWOT Analysis",
                    "paragraphs": [
                        {{
                            "title": "Strengths:"
                            "content": "\\n1. Great product\\n2. Strong Team"
                        }},
                        {{
                            "title": "Weaknesses:",
                            "content": "\\n1. Bad location\\n2. CEO low morale"
                        }},
                        {{
                            "title": "Opportunities:",
                            "content": "\\n1. Cyber data\\n2. Targeting new markets"
                        }},
                        {{
                            "title": "Threats:",
                            "content": "\\n1. Strong competition\\n2. Economic downturn"
                        }}
                    ]
                }}
            ]
        }}
        #### End of example
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
        llm=claude_haiku if haiku else claude_sonnet,
        prompt=string_to_report_prompt,
    )
    response = chain.invoke({"string": string})
    parsed_result = report_parser.parse(text=response["text"])
    # write raw string to object
    parsed_result.raw = string
    # add report style
    parsed_result.style = report_style
    return parsed_result


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
