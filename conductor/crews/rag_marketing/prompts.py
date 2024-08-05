"""
Turn each of the crew run into a section of a report.
"""
from conductor.reports.models import SectionV2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


REPORT_SECTION_PROMPT = """
You are a world class writer and you have been tasked with writing a section of a report.
Include all the information in the included context.
If valid URL sources are provided, include them at the end of the section. Else, leave the sources empty.
Each paragraph should be a minimum of {min_sentences} sentences and maximum of {max_sentences}, be sure to use the maximum if needed.
Be creative when beginning and ending paragraphs, avoid repetitive phrases such as "In conclusion" or "In summary".


Title:
<title>
{title}
If no title is provided, generate a title based on the context.
</title>

Style:
<style>
{style}
</style>

Tone:
<tone>
{tone}
</tone>

Point of view:
<point_of_view>
{point_of_view}
</point_of_view>

Context:
<context>
{context}
</context>

Format Instructions:
<format_instructions>
{format_instructions}
</format_instructions>
"""

section_parser = PydanticOutputParser(pydantic_object=SectionV2)

report_section_prompt = PromptTemplate(
    template=REPORT_SECTION_PROMPT,
    input_variables=[
        "title",
        "style",
        "tone",
        "context",
        "min_sentences",
        "max_sentences",
        "point_of_view",
    ],
    partial_variables={"format_instructions": section_parser.get_format_instructions()},
)
