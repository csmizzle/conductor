"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.reports.models import (
    Graph,
    RelationshipType,
    ImageSearchResult,
    ImageResult,
)
from conductor.chains.tools import ImageProcessor
from conductor.chains import relationships_to_image_query
from bs4 import BeautifulSoup
from datetime import datetime
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.zen import zenrows_client
from conductor.llms import openai_gpt_4o
from langchain_core.language_models.chat_models import BaseChatModel
import requests
import logging
from tqdm import tqdm
from typing import Union


logger = logging.getLogger(__name__)


# text data from websites
def ingest_webpage(url: str, limit: int = 50000, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
    response = None
    try:
        # get a created at timestamp
        created_at = datetime.now()
        # handle pdfs by passing for now
        if not url.endswith("pdf"):
            # use zenrows to get the text from the webpage
            params = dict(
                js_render="true",
                premium_proxy="true",
            )
            zen_response = zenrows_client.get(url, params=params, timeout=10)
            # process response and try with requests if not successful
            if not zen_response.ok:
                print(f"Zenrows Error: {zen_response.status_code}")
                print(f"Zenrows Error: {zen_response.text}")
                print("Sending request with requests instead ...")
                normal_response = requests.get(url, timeout=10, **kwargs)
                if not normal_response.ok:
                    print(f"Requests Error: {zen_response.status_code}")
                    print(f"Requests Error: {zen_response.text}")
                    normal_response.raise_for_status()
                else:
                    response = normal_response
            else:
                response = zen_response
            # process response if successful
            if response:
                # get text from response
                response_text = response.text
                # parse with BeautifulSoup
                soup = BeautifulSoup(response_text, "html.parser")
                # get text from soup
                text = soup.get_text(strip=True)
                # get the first limit characters
                text = text[:limit]
                # use the partition_text function
                return WebPage(
                    url=url, created_at=created_at, content=text, raw=response_text
                )
    except Exception as e:
        raise e


def url_to_db(url: str, client: ElasticsearchRetrieverClient, **kwargs) -> list[str]:
    """
    Ingest webpage from URL to Elasticsearch
    """
    # ingest webpage
    webpage = ingest_webpage(url, **kwargs)
    # insert document
    return client.create_insert_webpage_document(webpage)


# image data from urls
def describe_image_from_url(
    image_url: str, model: BaseChatModel, metadata: str = None, save_path: str = None
) -> SourcedImageDescription:
    """
    Ingest image from URL
    """
    try:
        # get a created at timestamp
        created_at = datetime.now()
        # get image content from url
        processor = ImageProcessor.from_url(
            image_url=image_url,
            model=model,
            metadata=metadata,
        )
        # describe image
        image_description = processor.describe()
        # save image to path if provided
        if save_path:
            processor.save_image(save_path)
        # return sourced image description
        return SourcedImageDescription(
            created_at=created_at,
            image_description=image_description,
            source=image_url,
            path=save_path,
        )
    except Exception as e:
        raise e


def describe_from_image_search_result(
    query: str,
    image_result: ImageSearchResult,
    model: BaseChatModel,
    save_path: str = None,
) -> SourcedImageDescription:
    """Describe an image from an image search result

    Args:
        query (str): Search query
        image_result (ImageSearchResult): image results
        model (BaseChatModel): Langchain model
        save_path (str, optional): path to save the files. Defaults to None.

    Raises:
        e: _description_

    Returns:
        SourcedImageDescription: Sourced image description
    """
    results = []
    try:
        # get a created at timestamp
        created_at = datetime.now()
        # get image content from url
        for image in image_result.results:
            processor = ImageProcessor.from_url(
                image_url=image.original_url,
                model=model,
                metadata=query + ";" + image.title,
            )
            # describe image
            image_description = processor.describe()
            # save image to path if provided
            if save_path:
                processor.save_image(save_path)
            # return sourced image description
            sourced_description = SourcedImageDescription(
                created_at=created_at,
                image_description=image_description,
                source=image_result.original_url,
                path=save_path,
            )
            results.append(sourced_description)
    except Exception as e:
        raise e
    return results


# now parse a list of queries to a list of image search results
def describe_from_image_search_results(
    image_results: list[ImageSearchResult],
    model: BaseChatModel,
    save_path: str = None,
) -> list[SourcedImageDescription]:
    """
    Describe images from image search results
    """
    results = {}
    for idx, _ in tqdm(enumerate(image_results), total=len(image_results)):
        try:
            # get a created at timestamp
            created_at = datetime.now()
            # get image content from url
            for image in image_results[idx].results:
                if image.original_url not in results:
                    results[image.original_url] = None
                else:
                    continue
                processor = ImageProcessor.from_url(
                    image_url=image.original_url,
                    model=model,
                    metadata=image_results[idx].query + ";" + image.title,
                )
                # describe image
                image_description = processor.describe()
                # save image to path if provided
                if save_path:
                    processor.save_image(save_path)
                # return sourced image description
                sourced_description = SourcedImageDescription(
                    created_at=created_at,
                    image_description=image_description,
                    source=image.original_url,
                    path=save_path,
                )
                results[image.original_url] = sourced_description
        except Exception as e:
            raise e
    return [results[key] for key in results]


def image_from_url_to_db(
    image_url: str,
    model: BaseChatModel,
    client: ElasticsearchRetrieverClient,
    metadata: str = None,
    save_path: str = None,
) -> Union[list[str], None]:
    """Ingest an image from a URL to Elasticsearch

    Args:
        image_url (str): URL of the image
        model (BaseChatModel): Chat model to describe the image
        client (ElasticsearchRetrieverClient): Elasticsearch client
        metadata (str, optional): Metadata from the image. Defaults to None.
        save_path (str, optional): Path to save the image to. Defaults to None.

    Returns:
        list[str]: _description_
    """
    try:
        # ingest image
        image = describe_image_from_url(
            image_url=image_url,
            model=model,
            metadata=metadata,
            save_path=save_path,
        )
        # insert document
        return client.create_insert_image_document(image)
    except Exception as e:
        print(f"Error uploading image {image_url} to db")
        print(e)


def queries_to_image_results(
    search_queries: list[str], n_images: int = 1
) -> list[ImageSearchResult]:
    """
    Convert queries to image search results
    """
    # collect searches from SerpAPI
    image_results = []
    for query in search_queries:
        image_result = ImageSearchResult(
            query=query["search_parameters"]["q"]
        )  # map raw query to image result
        # collect n results from results
        for idx in range(n_images):
            if "images_results" in query:
                image_result.results.append(
                    ImageResult(
                        original_url=query["images_results"][idx]["original"],
                        title=query["images_results"][idx]["title"],
                    )
                )
                image_results.append(image_result)
            else:
                logger.info(
                    f"No image results found for {query["search_parameters"]["q"]}"
                )
    return image_results


def insert_image_urls_to_db(
    image_results: list[ImageSearchResult],
    client: ElasticsearchRetrieverClient,
    save_path: str = None,
) -> list[str]:
    added_documents = []
    for result in tqdm(image_results):
        logger.info("Parsing image entry ...")
        for entry in result.results:
            # process image
            document = image_from_url_to_db(
                image_url=entry.original_url,
                model=openai_gpt_4o,
                client=client,
                metadata=result.query
                + "; "
                + entry.title,  # append the query and title for metadata
                save_path=save_path if save_path else None,
            )
            added_documents.extend(document)
    return added_documents


def ingest_images_from_graph(
    graph: Graph,
    api_key: str,
    relationship_types: list[RelationshipType],
    client: ElasticsearchRetrieverClient,
    n_images: int = 1,
    save_path: str = None,
) -> list[str]:
    """Ingest images from a Graph

    Args:
        graph (Graph): Extracted graph
        api_key (str): SerpAPI credentials
        relationship_types (list[RelationshipType]): Relationship types to extract
        client (ElasticsearchRetrieverClient): Elasticsearch client
        n_images (int, optional): Number of images to pull from SerpAPI. Defaults to 1.
        save_path (str, optional): Path to save images to. Defaults to None.

    Returns:
        list[str]: Confirmation of images being added
    """
    search_queries = relationships_to_image_query(
        graph=graph,
        api_key=api_key,
        relationship_types=relationship_types,
    )
    # collect searches from SerpAPI
    image_results = queries_to_image_results(search_queries, n_images)
    # describe images
    added_documents = insert_image_urls_to_db(
        image_results=image_results,
        client=client,
        save_path=save_path,
    )
    return added_documents
