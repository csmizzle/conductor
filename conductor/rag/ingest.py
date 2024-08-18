"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage, SourcedImageDescription
from conductor.chains.tools import ImageProcessor
from bs4 import BeautifulSoup
from datetime import datetime
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.rag.client import zenrows_client
from langchain_core.language_models.chat_models import BaseChatModel


# text data from websites
def ingest_webpage(url: str, limit: int = 50000, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
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
            response = zenrows_client.get(
                url,
                params=params,
            )
            # process response
            if not response.ok:
                print(f"Error: {response.status_code}")
                print(f"Error: {response.text}")
                response.raise_for_status()
            else:
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
def ingest_image_from_url(
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


def image_from_url_to_db(
    image_url: str,
    model: BaseChatModel,
    client: ElasticsearchRetrieverClient,
    metadata: str = None,
    save_path: str = None,
) -> list[str]:
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
    # ingest image
    image = ingest_image_from_url(
        image_url=image_url,
        model=model,
        metadata=metadata,
        save_path=save_path,
    )
    # insert document
    return client.create_insert_image_document(image)
