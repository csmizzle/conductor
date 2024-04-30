"""
A custom spider that uses LLMs to process the HTML content of a web page into a summary.
"""
from typing import Generator
from scrapy import Spider
from scrapy.responsetypes import Response
from bs4 import BeautifulSoup
from conductor.collect.items import SummaryItem
from conductor.utils import clean_string


class SummarySpider(Spider):
    """
    Scrapes and summarizes a web page.
    """

    name = "summary_spider"
    custom_settings = {
        "ITEM_PIPELINES": {"conductor.collect.pipelines.SummaryItemPipeline": 300}
    }

    def __init__(self, urls: list[str], *args, **kwargs):
        """
        Initialize the spider with a list of URLs to scrape.
        """
        super().__init__(*args, **kwargs)
        self.start_urls = urls

    def parse(self, response: Response) -> Generator[SummaryItem, None, None]:
        """Parse a raw response and generate a summary.

        Args:
            response (Response): The raw response from the web page.

        Yields:
            Generator[SummaryItem]: summary item
        """
        self.logger.info("SummarySpider is parsing %s...", response)
        raw = response.body
        text = BeautifulSoup(raw, "html.parser").get_text()
        cleaned_text = clean_string(text)
        yield {
            "url": response.url,
            "content": cleaned_text,
        }
