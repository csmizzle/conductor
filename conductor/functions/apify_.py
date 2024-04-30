"""
Functions that interact with the Apify spiders
"""
from conductor.collect.spiders.summary import SummarySpider
from scrapy.crawler import CrawlerProcess
from apify import Actor
from apify.scrapy.utils import apply_apify_settings
import asyncio
from twisted.internet import asyncioreactor
import nest_asyncio

nest_asyncio.apply()
asyncioreactor.install()


async def summarize_urls(urls: list[str], stop: bool = True) -> None:
    """
    Scrape and summarize a web page.
    """
    async with Actor:
        actor_input = await Actor.get_input() or {}
        proxy_config = actor_input.get("proxyConfiguration")
        # Add start URL to the request queue
        print("Creating URL queue ...")
        rq = await Actor.open_request_queue()
        for url in urls:
            await rq.add_request(request={"url": url, "method": "GET"})
        # Apply Apify settings, it will override the Scrapy project settings
        settings = apply_apify_settings(proxy_config=proxy_config)
        # Execute the spider using Scrapy CrawlerProcess
        print("Starting CrawlerProcess ...")
        process = CrawlerProcess(settings, install_root_handler=False)
        print("Starting crawl ...")
        process.crawl(SummarySpider, urls=urls)
        print("Starting process ...")
        process.start(stop_after_crawl=stop)


def run_summarize_urls(urls: list[str], stop: bool = True) -> None:
    return asyncio.run(summarize_urls(urls=urls, stop=stop))
