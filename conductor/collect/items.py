"""
Scrapy item models module

This module defines Scrapy item models for scraped data. Items represent structured data
extracted by spiders.

For detailed information on creating and utilizing items, refer to the official documentation:
https://docs.scrapy.org/en/latest/topics/items.html
"""

from scrapy import Field, Item


class SummaryItem(Item):
    """
    A summary containing the url, raw html, and processed summary
    """

    task_id = Field()
    job_id = Field()
    url = Field()
    content = Field()
    summary = Field()
