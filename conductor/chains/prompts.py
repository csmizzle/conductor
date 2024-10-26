"""
Prompts for entity extraction
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from conductor.reports.models import Graph, Timeline, QueryMatch


# Entity extraction
ENTITY_EXTRACTION_PROMPT = """
You are a state of the art entity extraction and relation extraction model.
Your goal is to extract entities and relationships from a given text and create a relational graph.
First, evaluate the entity types and relationship types you can extract.
Second, evaluate the provided text and extract entities.
Third, evaluate the provided text and extract relationships between the entities you extracted.
Fourth, deduplicate the entities and relationships you extracted and merge similar entities into one.
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
# graph_retry_parser = OutputFixingParser(parser=graph_parser)
graph_extraction_prompt = PromptTemplate(
    template=ENTITY_EXTRACTION_PROMPT,
    input_variables=[
        "entity_types",
        "relationship_types",
        "text",
    ],
    partial_variables={"format_instructions": graph_parser.get_format_instructions()},
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
