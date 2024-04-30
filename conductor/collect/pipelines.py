"""
Scrapy item pipelines module

This module defines Scrapy item pipelines for scraped data. Item pipelines are processing components
that handle the scraped items, typically used for cleaning, validating, and persisting data.

For detailed information on creating and utilizing item pipelines, refer to the official documentation:
http://doc.scrapy.org/en/latest/topics/item-pipeline.html
"""

from scrapy import Spider
from conductor.collect.items import SummaryItem
from conductor.database.aws import upload_dict_to_s3
from conductor.chains import get_parsed_html_summary
import os
import uuid


class SummaryItemPipeline:
    """
    This item pipeline defines processing steps for SummaryItem objects scraped by spiders.
    """

    def process_item(self, item: dict, spider: Spider) -> SummaryItem:
        # Do something with the item here, such as cleaning it or persisting it to a database
        print("Running pipeline ...")
        summary = get_parsed_html_summary(item["content"])
        item["summary"] = summary.summary
        upload_dict_to_s3(
            data=dict(item),
            bucket=os.getenv("APIFY_S3_BUCKET"),
            key=f"{str(uuid.uuid4())}_summary.json",
        )
        return SummaryItem(
            url=item["url"], content=item["content"], summary=item["summary"]
        )
