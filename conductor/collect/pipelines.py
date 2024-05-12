"""
Scrapy item pipelines module

This module defines Scrapy item pipelines for scraped data. Item pipelines are processing components
that handle the scraped items, typically used for cleaning, validating, and persisting data.

For detailed information on creating and utilizing item pipelines, refer to the official documentation:
http://doc.scrapy.org/en/latest/topics/item-pipeline.html
"""

from scrapy import Spider
from conductor.collect.items import SummaryItem
from conductor.chains import get_parsed_html_summary
import uuid


class SummaryItemPipeline:
    """
    This item pipeline defines processing steps for SummaryItem objects scraped by spiders.
    """

    def process_item(self, item: dict, spider: Spider) -> SummaryItem:
        # Do something with the item here, such as cleaning it or persisting it to a database
        print("Running pipeline ...")
        job_id = str(uuid.uuid4())
        summary = get_parsed_html_summary(item["content"])
        item["summary"] = summary.summary
        return SummaryItem(
            task_id=item["task_id"],
            job_id=job_id,
            url=item["url"],
            content=item["content"],
            summary=item["summary"],
        )
