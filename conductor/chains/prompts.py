"""
Prompts for entity extraction
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from conductor.reports.models import Graph, Timeline, QueryMatch
from conductor.chains.models import SyntheticDocuments
from conductor.reports.models import SectionV2


# Entity extraction
ENTITY_EXTRACTION_PROMPT = """
You are a state of the art entity extraction and relation extraction model.
Your goal is to extract entities and relationships from a given text and create a relational graph.
First, evaluate the entity types and relationship types you can extract.
Second, evaluate the provided text and extract entities.
Third, evaluate the provided text and extract relationships between the entities you extracted.
Fourth, deduplicate the entities and relationships you extracted.
For each entity, provide a reason for why you think the entity is of that type and provide the other names provided if there was any deduplication.
For each relationship, provide a reason for why you think the relationship exists.
Finally, return the extracted entities and relationships in the provided JSON format.

<entity_types>
{entity_types}
</entity_types>

<relationship_types>
{relationship_types}
</relationship_types>

<text>
{text}
</text>

<format_instructions>
{format_instructions}
</format_instructions>
"""

graph_parser = PydanticOutputParser(pydantic_object=Graph)
graph_retry_parser = OutputFixingParser(parser=graph_parser)
graph_extraction_prompt = PromptTemplate(
    template=ENTITY_EXTRACTION_PROMPT,
    input_variables=[
        "entity_types",
        "relationship_types",
        "text",
    ],
    partial_variables={
        "format_instructions": graph_retry_parser.get_format_instructions()
    },
)


# Timeline extraction
TIMELINE_EXTRACTION_PROMPT = """
You are a state of the art timeline extraction model.
Your goal is to extract events from a given text and create a timeline.
First, evaluate the provided text and extract events.
The events need to have a year and a short description.
If the event has a month and day, include them as well.
Finally, return the extracted events in the provided JSON format.

<text>
{text}
</text>

<format_instructions>
{format_instructions}
</format_instructions>
"""

timeline_parser = PydanticOutputParser(pydantic_object=Timeline)


timeline_extraction_prompt = PromptTemplate(
    template=TIMELINE_EXTRACTION_PROMPT,
    input_variables=[
        "text",
    ],
    partial_variables={
        "format_instructions": timeline_parser.get_format_instructions()
    },
)


# query to paragraph matching
QUERY_TO_PARAGRAPH_MATCHING_PROMPT = """
You are a world-class model for matching search queries to paragraph text, making sure that the query text is relevant to the paragraph text.
Your goal is to evaluate the provided search query and paragraph text and determine if the search query is relevant to the paragraph text.
You have three options:
- RELEVANT: The search query is relevant to the paragraph text.
- NOT_RELEVANT: The search query is not relevant to the paragraph text.
- UNSURE: You are unsure if the search query is relevant to the paragraph text.
Finally, return the evaluations in the provided JSON format.

<search_query>
{search_query}
</search_query>

<paragraph_text>
{paragraph_text}
</paragraph_text>

<format_instructions>
{format_instructions}
</format_instructions>
"""

query_matcher_parser = PydanticOutputParser(pydantic_object=QueryMatch)

query_to_paragraph_matching_prompt = PromptTemplate(
    template=QUERY_TO_PARAGRAPH_MATCHING_PROMPT,
    input_variables=[
        "search_query",
        "paragraph_text",
    ],
    partial_variables={
        "format_instructions": query_matcher_parser.get_format_instructions()
    },
)


CAPTION_PROMPT = """
You are a world class short caption writer.
Write short captions for reports that will go in professional reports.
Use the search query and image title fields to write short, informational image captions.
The caption should be no longer than 6 words.

<search_query>
{search_query}
</search_query>

<image_title>
{image_title}
</image_title>
"""

caption_prompt = PromptTemplate(
    template=CAPTION_PROMPT, input_variables=["search_query", "image_title"]
)


HYDE_QUERY_GENERATION_PROMPT = """
You are a world class synthetic document query generator.
Your goal is to generate larger synthetic documents that will retrieve the most relevant documents from the vector database.
First, evaluate the provided context, objective, and today's date.
Use todays date to ground the information in recent data by using the date in the query.
Second, generate synthetic documents that will retrieve the most relevant and recent documents from a vector store.
Each document should have a minimum of {n_sentences} sentences.
Finally, return the generated documents in the provided JSON format.

<date>
{date}
</date>

<context>
{context}
</context>

<n_sentences>
{n_sentences}
</n_sentences>

<objective>
{objective}
</objective>

<n_documents>
{n_documents}
</n_documents>

<format_instructions>
{format_instructions}
</format_instructions>
"""


hyde_parser = PydanticOutputParser(pydantic_object=SyntheticDocuments)
hyde_fixing_parser = OutputFixingParser(parser=hyde_parser)
hyde_prompt = PromptTemplate(
    template=HYDE_QUERY_GENERATION_PROMPT,
    input_variables=["date", "context", "objective", "n_documents", "n_sentences"],
    partial_variables={
        "format_instructions": hyde_fixing_parser.get_format_instructions()
    },
)


SOURCED_SECTION_WRITER_PROMPT = """
You are a world class writer and you have been tasked with writing a section of a report.
Use as many paragraphs as needed to cover the context provided, be thorough, detailed, and include all relevant information.
Pull all the information from the context provided and ensure that all information is accurate and well-sourced.

First, use the provided title, style, tone, and point of view to guide your writing.
Second, use the previous sections as a guide for reducing repetition and ensuring consistency.
- If previous sections are not provided, ignore this step.
Third, write the section with the provided context with their sources.
- Each paragraph should be a minimum of {min_sentences} sentences and maximum of {max_sentences}, be sure to use the maximum if needed.
- Each sentence should sourced with at least one source, in the form of footnotes.
    - Sources should be in the form of urls.
    - Sources should be unique and not repeated.
    - Each url should only map to a single footnote number.
        - If a source is used twice, it should use the same footnote number.
    - If there is not a url and instead a description of an image, use the description in the source.
    - If there are multiple sources for a sentence, separate them like this: [1][2].
        - Parse these footnotes into the correct JSON format.
- Be creative when beginning and ending paragraphs, avoid repetitive phrases such as "In conclusion" or "In summary".
- Avoid repeating the same information in different ways, repetitive paragraph openers, and repetitive sentence structures.
Thirdly, ensure that all footnotes are:
- Mapped to the correct sentence.
- One URL to one footnote.
    - If you find a URL that is repeated, replace the repeated URL with the earlier footnote number and remove the duplicate later footnote number.
Finally, return the section in the provided JSON format.


Example:

Context: Thomson Reuters Special Services was founded in 2018 by John Doe. The company specializes in providing security services to high net worth individuals and corporations. The company has a team of highly trained security professionals who have experience in the military and law enforcement. Source: trssllc.com.
Output: Thomson Reuters Special Services was founded in 2018 by John Doe.[1] The company specializes in providing security services to high net worth individuals and corporations.[1] The company has a team of highly trained security professionals who have experience in the military and law enforcement.[1]

End of Example


Previous Sections:
<previous_sections>
{previous_sections}
</previous_sections>


Title:
<title>
{title}
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


sourced_section_parser = PydanticOutputParser(pydantic_object=SectionV2)
sourced_section_fixing_parser = OutputFixingParser(parser=sourced_section_parser)
sourced_section_writer_prompt = PromptTemplate(
    template=SOURCED_SECTION_WRITER_PROMPT,
    input_variables=[
        "title",
        "style",
        "tone",
        "point_of_view",
        "context",
        "min_sentences",
        "max_sentences",
        "previous_sections",
    ],
    partial_variables={
        "format_instructions": sourced_section_fixing_parser.get_format_instructions()
    },
)
