"""
Ingest of raw data into Pydantic model
"""
from conductor.rag.models import WebPage
from docling.document_converter import DocumentConverter
from bs4 import BeautifulSoup
from datetime import datetime
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.zen import zenrows_client
import requests
from typing import Union
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed


def make_request(url: str, **kwargs) -> Union[requests.Response, None]:
    zen_response = None
    response = None
    # use zenrows to get the text from the webpage
    params = dict(
        js_render="true",
        premium_proxy="true",
    )
    try:
        zen_response = zenrows_client.get(url, params=params, timeout=20)
    except requests.exceptions.ReadTimeout:
        logger.error(f"Zenrows Timeout Error: {url}")
    # process response and try with requests if not successful
    if not zen_response or not zen_response.ok:
        if zen_response:
            logger.error(f"Zenrows Error: {zen_response.status_code}")
            logger.error(f"Zenrows Error: {zen_response.text}")
        logger.error("Sending request with requests instead ...")
        normal_response = requests.get(url, **kwargs)
        if not normal_response.ok:
            logger.error(f"Requests Error: {normal_response.status_code}")
            logger.error(f"Requests Error: {normal_response.text}")
            normal_response.raise_for_status()
        else:
            response = normal_response
    else:
        response = zen_response
    return response


# text data from websites
def ingest_webpage(
    url: str, limit: int = 50000, pdf_page_limit: int = 10, **kwargs
) -> WebPage:
    """
    Ingest webpage from URL
    """
    response = None
    try:
        # get a created at timestamp
        created_at = datetime.now()
        # make request
        response = make_request(url, **kwargs)
        # process response if successful
        if response:
            if response.headers.get("content-type") != "application/pdf":
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
            # ingest pdfs
            else:
                logger.info(f"Ingesting PDF from {url} ...")
                # docling document extractor from temp file
                converter = DocumentConverter()
                results = converter.convert(
                    source=url,
                    max_num_pages=pdf_page_limit,
                )
                markdown = results.document.export_to_markdown()
                return WebPage(
                    url=url,
                    created_at=created_at,
                    content=markdown,
                    raw=results.document.export_to_text(),
                )
    except Exception as e:
        logger.error(f"Error ingesting {url}")
        raise e


def url_to_db(url: str, client: ElasticsearchRetrieverClient, **kwargs) -> list[str]:
    """
    Ingest webpage from URL to Elasticsearch
    """
    # ingest webpage
    webpage = ingest_webpage(url, **kwargs)
    # insert document
    return client.create_insert_webpage_document(webpage)


def ingest_with_ids(
    client: ElasticsearchRetrieverClient,
    url: str,
    headers: dict = None,
    cookies: dict = None,
) -> dict[str, list[str]]:
    logger.info(f"Ingesting data for {url} ...")
    existing_documents = client.find_documents_by_url(url=url)
    if len(existing_documents) > 0:
        logger.info(f"Documents already exists for {url}, returning document ids")
        document_ids = [doc["_id"] for doc in existing_documents]
        return {url: document_ids}
    else:
        try:
            webpage = url_to_db(
                url=url, client=client, headers=headers, cookies=cookies, timeout=10
            )
            return {url: webpage}
        except Exception:
            logger.error(f"Error uploading {url} to db")


def parallel_ingest_with_ids(
    urls, client, headers=None, cookies=None
) -> dict[str, list[str]]:
    """
    Run ingest_with_ids in parallel
    """
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for url in urls:
            futures.append(
                executor.submit(
                    ingest_with_ids,
                    client=client,
                    url=url,
                    headers=headers,
                    cookies=cookies,
                )
            )
        for future in futures:
            result = future.result()
            if result:
                results.update(result)
    return results


def ingest(
    client: ElasticsearchRetrieverClient,
    url: str,
    headers: dict = None,
    cookies: dict = None,
):
    logger.info(f"Ingesting data for {url} ...")
    existing_document = client.find_document_by_url(url=url)
    if existing_document["hits"]["total"]["value"] > 0:
        return "Document already exists in the vector database"
    else:
        webpage = url_to_db(
            url=url, client=client, headers=headers, cookies=cookies, timeout=10
        )
        return f"New documents added: {', '.join(webpage)}"


# parallelized ingest function
def parallel_ingest(urls, client, headers=None, cookies=None):
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(ingest, client, url, headers, cookies): url for url in urls
        }
        results = []
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"Error processing: {url}")
                logger.error(f"Error: {url} --> {e}")
    return results
