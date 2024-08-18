"""
Prompts for entity extraction
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from conductor.chains.models import Graph, Timeline


# Entity extraction
ENTITY_EXTRACTION_PROMPT = """
You are a state of the art entity extraction and relation extraction model.
Your goal is to extract entities and relationships from a given text and create a relational graph.
First, evaluate the entity types and relationship types you can extract.
Second, evaluate the provided text and extract entities.
Third, evaluate the provided text and extract relationships between the entities you extracted.
For each entity, provide a reason for why you think the entity is of that type.
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
