"""
Conductor Report Outputs
- PDF Report
"""
import pdfkit
from langchain.prompts import PromptTemplate
from conductor.reports.models import ParsedReport, Report
from conductor.llms import openai_gpt_4o
from textwrap import dedent
import tempfile
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.llm import LLMChain
from langsmith import traceable
from docx import Document
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
        report_title=report.report.title,
        report_description=report.report.description,
        report_sections=[section.dict() for section in report.report.sections],
    )
    return html.unescape(output_from_parsed_template)


@traceable
def string_to_report(
    string: str,
) -> ParsedReport:
    """
    Generate a report object from a string
    """
    report_parser = PydanticOutputParser(pydantic_object=ParsedReport)
    string_to_report_prompt = PromptTemplate(
        input_variables=["string"],
        template=dedent(
            """
        Convert the following string to a Report JSON object.
        Do not modify the data, just parse it into the JSON object.
        Sections should be marked by a number and a title.
        The key sections are Overview, Market Analysis, SWOT Analysis, Competitors, and Sources.

        Each Section should be be broken down into paragraphs.
        Include newlines in individual sections to that the sections maintain the original document structure when parsed into JSON.
        Always include all the sections and paragraphs from the original report.

        #### Example:
        1. Overview
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

        2. Market Analysis
        Market:
        The market for widgets is growing.

        TAM/SAM/SOM:
        The TAM for widgets is $1 billion.

        3. SWOT Analysis
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

        4. Competitors
        Competitor 1
        Competitor 1 is a competitor of Acme Corp.

        Competitor 2
        Competitor 2 is a competitor of Acme Corp.

        5. Sources
        - https://acmecorp.com
        - https://acmecorp.com/about

        #### Example JSON Object:
        {{
            "title": "Acme Corp Report",
            "description": "This is a report on Acme Corp.",
            "sections": [
                {{
                    "title": "1. Overview",
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
                    "title": "2. Market Analysis",
                    "paragraphs": [
                        {{
                            "title": "Market:"
                            "content": "\\nThe market for widgets is growing."
                        }},
                        {{
                            "title": "TAM/SAM/SOM:"
                            "content": "\\nThe TAM for widgets is $1 billion."
                        }}
                    ]
                }},
                {{
                    "title": "3. SWOT Analysis",
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
                }},
                {{
                    "title": "4. Competitors",
                    "paragraphs": [
                        {{
                            "title": "Competitor 1"
                            "content": "\\nCompetitor 1 is a competitor of Acme Corp."
                        }},
                        {{
                            "title": "Competitor 2"
                            "content": "\\nCompetitor 2 is a competitor of Acme Corp."
                        }}
                    ]
                }},
                {{
                    "title": "5. Sources",
                    "paragraphs": [
                        {{
                            "title": "Links"
                            "content": "\\n- https://acmecorp.com\\n- https://acmecorp.com/about"
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
        llm=openai_gpt_4o,
        prompt=string_to_report_prompt,
    )
    response = chain.invoke({"string": string})
    parsed_result = report_parser.parse(text=response["text"])
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


def report_to_docx(report: Report) -> Document:
    """
    Convert a report to a DOCX
    """
    document = Document()
    document.add_heading(report.report.title, level=1)
    for section in report.report.sections:
        document.add_heading(section.title, level=2)
        for paragraph in section.paragraphs:
            if paragraph.title and paragraph.title != "":
                document.add_heading(paragraph.title, level=3)
            document.add_paragraph(paragraph.content)
    return document
