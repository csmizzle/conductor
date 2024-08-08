"""
Ingest of raw data into Pydantic model
"""
import asyncio
from conductor.rag.models import WebPage
from bs4 import BeautifulSoup
import requests
from time import sleep
from datetime import datetime
from pyppeteer import launch
from conductor.rag.client import ElasticsearchRetrieverClient


def fetch_webpage_screenshot(url: str, screenshot_path: str, **kwargs) -> str:
    """
    Fetch the content of a webpage using pyppeteer synchronously.
    """

    async def get_content():
        browser = await launch(headless=True)
        page = await browser.newPage()
        page.setUserAgent(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36"
        )
        await page.goto(url)
        sleep(5)
        content = await page.content()
        # full_height = await page.evaluate('document.body.scrollHeight')
        # # Set the viewport height to the full height of the page
        await page.setViewport({"width": 1280})
        await page.screenshot({"path": screenshot_path})
        await browser.close()
        return content

    return asyncio.run(get_content())


def ingest_webpage(url: str, **kwargs) -> WebPage:
    """
    Ingest webpage from URL
    """
    # get a created at timestamp
    created_at = datetime.now()
    # handle pdfs by passing for now
    if not url.endswith("pdf"):
        # use requests
        response = requests.get(url, **kwargs)
        # process response
        if not response.ok:
            response.raise_for_status()
        else:
            # get text from response
            response_text = response.text
            # parse with BeautifulSoup
            soup = BeautifulSoup(response_text, "html.parser")
            # get text from soup
            text = soup.get_text(strip=True)
            # use the partition_text function
            return WebPage(
                url=url, created_at=created_at, content=text, raw=response_text
            )
    else:
        return WebPage(
            url=url, content="Unable to parse PDF", raw="", created_at=created_at
        )


def url_to_db(url: str, client: ElasticsearchRetrieverClient, **kwargs) -> list[str]:
    """
    Ingest webpage from URL to Elasticsearch
    """
    # ingest webpage
    webpage = ingest_webpage(url, **kwargs)
    # insert document
    return client.create_insert_webpage_document(webpage)
