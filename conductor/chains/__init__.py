from conductor.chains import prompts
from conductor.chains.tools import image_search
from datetime import date
from conductor.chains.models import SyntheticDocuments
from conductor.llms import openai_gpt_4o
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.reports.models import (
    Graph,
    EntityType,
    RelationshipType,
    Timeline,
    ImageSearchResult,
    QueryMatch,
    ReportV2,
    SectionV2,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tqdm import tqdm

graph_chain = (
    prompts.graph_extraction_prompt | openai_gpt_4o | prompts.graph_retry_parser
)
timeline_chain = (
    prompts.timeline_extraction_prompt | openai_gpt_4o | prompts.timeline_parser
)
query_match_chain = (
    prompts.query_to_paragraph_matching_prompt
    | openai_gpt_4o
    | prompts.query_matcher_parser
)
caption_chain = prompts.caption_prompt | openai_gpt_4o | StrOutputParser()
hyde_chain = prompts.hyde_prompt | openai_gpt_4o | prompts.hyde_parser
sourced_section_writer_chain = (
    prompts.sourced_section_writer_prompt
    | openai_gpt_4o
    | prompts.sourced_section_fixing_parser
)


def run_graph_chain(
    entity_types: list[str], relationship_types: list[str], text: str
) -> Graph:
    """Extract entities and relationships from a given text and create a relational graph.

    Args:
        entity_types (list[str]): entity types to extract
        relationship_types (list[str]): relationship types to extract
        text (str): text to extract entities and relationships from

    Returns:
        Graph: graph object containing the extracted entities and relationships
    """
    return graph_chain.invoke(
        dict(
            entity_types=entity_types,
            relationship_types=relationship_types,
            text=text,
        )
    )


def run_set_graph_chain(text: str) -> Graph:
    """Extract entities and relationships from a given text and create a relational graph.

    Args:
        text (str): text to extract entities and relationships from

    Returns:
        Graph: graph object containing the extracted entities and relationships
    """
    return run_graph_chain(
        text=text,
        entity_types=[enum.value for enum in EntityType],
        relationship_types=[enum.value for enum in RelationshipType],
    )


def run_timeline_chain(text: str) -> Timeline:
    """Extract events from a given text and create a timeline.

    Args:
        text (str): text to extract events from

    Returns:
        Timeline: timeline object containing the extracted events
    """
    return timeline_chain.invoke(
        dict(
            text=text,
        )
    )


def run_query_match_chain(search_query: str, text: str) -> QueryMatch:
    """Match a search query to a given text.

    Args:
        search_query (str): search query to match
        text (str): text to match the search query to

    Returns:
        QueryMatch: query match object containing the search query and the matched text
    """
    return query_match_chain.invoke(
        dict(
            search_query=search_query,
            paragraph_text=text,
        )
    )


def run_create_caption_chain(image_title: str, search_query: str) -> str:
    return caption_chain.invoke(
        dict(image_title=image_title, search_query=search_query)
    )


def run_hyde_generation_chain(
    context: str, objective: str, n_documents: int = 5, n_sentences: int = 5
) -> SyntheticDocuments:
    return hyde_chain.invoke(
        dict(
            date=date.today().strftime(
                "%Y-%m-%d"
            ),  # ground LLM in today's date for recent information
            context=context,
            objective=objective,
            n_documents=n_documents,
            n_sentences=n_sentences,
        )
    )


# relationship to image search
def relationships_to_image_query(
    graph: Graph,
    api_key: str,
    relationship_types: list[RelationshipType] = None,
) -> list[dict]:
    """
    Converts a relationship to an image search.

    Args:
        graph (Graph): The graph to convert to an image search.
        api_key (str): The API key for the image search.
        relationship_types (list[RelationshipType]): The relationship types to convert to an image search. If None, all relationships are converted.

    Returns:
        str: The image search.
    """
    # iterate through graph and collect relations
    searches = set()
    for relationship in graph.relationships:
        # filter relationships based on relationship types
        if relationship_types:
            if relationship.type in relationship_types:
                # concat source and target and add to searches
                searches.add(relationship.source.name + " " + relationship.target.name)
        # add all relationships
        else:
            searches.add(relationship.source.name + " " + relationship.target.name)
    # iterate through searches and create image search
    results = []
    for search in tqdm(searches):
        results.append(
            image_search(
                query=search,
                api_key=api_key,
            )
        )
    return results


def match_queries_to_paragraphs(
    image_search_results: list[ImageSearchResult],
    sections_filter: list[str],
    report: ReportV2,
) -> ReportV2:
    """
    Assign photos to paragraphs based on image search results.
    There should not be:
    - Duplicate photos in the report
    - Multiple photos in a paragraph

    Args:
        image_search_results (list[ImageSearchResult]): _description_
        sections_filter (list[str]): _description_
        report (ReportV2): _description_

    Returns:
        ReportV2: _description_
    """
    section_paragraph_matches = []
    matched_queries = []
    matched_images = []
    # iterate through paragraphs and match image search results
    for idx0 in range(len(report.report.sections)):
        if report.report.sections[idx0].title in sections_filter:
            for idx1 in range(len(report.report.sections[idx0].paragraphs)):
                section_paragraph_key = f"{idx0}-{idx1}"
                paragraph_text = " ".join(
                    report.report.sections[idx0].paragraphs[idx1].sentences
                )
                # run match chain on paragraph text with image search results query
                for idx2 in range(len(image_search_results)):
                    # ensure the image search result has not been matched
                    if idx2 not in matched_queries:
                        search_query = image_search_results[idx2].query
                        image_match = run_query_match_chain(
                            search_query=search_query, text=paragraph_text
                        )
                        # if the search query matches the paragraph text, assign the image to the paragraph
                        if image_match.determination == "RELEVANT":
                            if section_paragraph_key not in section_paragraph_matches:
                                if (
                                    image_search_results[idx2].query
                                    not in matched_images
                                ):
                                    print(
                                        f"Matched '{search_query}' to paragraph text, assigning image..."
                                    )
                                    matched_queries.append(idx2)
                                    if len(image_search_results[idx2].results) > 0:
                                        matched_images.append(
                                            image_search_results[idx2]
                                            .results[0]
                                            .original_url
                                        )
                                        report.report.sections[idx0].paragraphs[
                                            idx1
                                        ].images = image_search_results[idx2]
                                        section_paragraph_matches.append(
                                            section_paragraph_key
                                        )
                                    else:
                                        print("No image results found, skipping...")
                                        continue
                                else:
                                    print(
                                        "Image URL already assigned to paragraph, skipping..."
                                    )
                                    continue
                            else:
                                print(
                                    "Images already assigned to paragraph, skipping..."
                                )
                                continue
                        else:
                            print(
                                f"Search query '{search_query}' did not match paragraph text, skipping..."
                            )
                            continue
                    else:
                        print(
                            f"Image search result {image_search_results[idx2].query} already matched, skipping..."
                        )
                        continue
    # return report with image search results
    return report


def run_hyde_search(
    context: str,
    objective: str,
    retriever: ElasticsearchRetrieverClient,
    n_documents: int = 5,
    n_sentences: int = 5,
    k: int = 4,
    fetch_k: int = 50,
    unique: bool = True,
) -> list[Document]:
    """Run the HYDE search chain and get the most relevant documents.

    Args:
        context (str): Query context.
        objective (str): Objective of the query.
        retriever (ElasticsearchRetrieverClient): Elasticsearch retriever client.
        n_documents (int, optional): Number of synthetic documents. Defaults to 5.
        n_sentences (int, optional): Number of sentences in a synthetic document. Defaults to 5.
        unique (bool, optional): Return unique documents. Defaults to True.

    Returns:
        list[Document]: List of relevant documents.
    """
    results = []
    if unique:
        urls = []
    synthetic_documents = run_hyde_generation_chain(
        context=context,
        objective=objective,
        n_documents=n_documents,
        n_sentences=n_sentences,
    )
    for document in synthetic_documents.documents:
        result = retriever.store.similarity_search(
            query=document,
            k=k,
            fetch_k=fetch_k,
        )
        if unique:
            # filter out duplicate urls if unique is True
            for document in result:
                if document.metadata["url"] not in urls:
                    urls.append(document.metadata["url"])
                    results.append(document)
                else:
                    continue
        else:
            results.extend(result)
    return results


def run_sourced_section_chain(
    title: str,
    style: str,
    tone: str,
    point_of_view: str,
    context: str,
    min_sentences: int = 3,
    max_sentences: int = 5,
    sections: list[str] = None,
) -> SectionV2:
    """Run the sourced section chain and get the most relevant documents.

    Args:
        title (str): Title of the section.
        style (str): Style of the section.
        tone (str): Tone of the section.
        point_of_view (str): Point of view of the section.
        context (str): Query context.
        min_sentences (int, optional): Minimum number of sentences in a paragraph. Defaults to 3.
        max_sentences (int, optional): Maximum number of sentences in a paragraph. Defaults to 5.
        sections (list[str], optional): Previous sections. Defaults to None.

    Returns:
        SectionV2: Section with paragraphs and sources.
    """
    return sourced_section_writer_chain.invoke(
        dict(
            title=title,
            style=style,
            tone=tone,
            point_of_view=point_of_view,
            context=context,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            previous_sections=sections,
        )
    )
