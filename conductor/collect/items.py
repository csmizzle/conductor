"""
Scrapy item models module

This module defines Scrapy item models for scraped data. Items represent structured data
extracted by spiders.

For detailed information on creating and utilizing items, refer to the official documentation:
https://docs.scrapy.org/en/latest/topics/items.html
"""

from scrapy import Field, Item


class TitleItem(Item):
    """
    Represents a title item scraped from a web page.
    """

    url = Field()
    title = Field()


class SummaryItem(Item):
    """
    A summary containing the url, raw html, and processed summary
    """

    url = Field()
    raw = Field()
    summary = Field()
