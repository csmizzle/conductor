"""
Functions that interact with the Apify spiders
"""
from conductor.collect.spiders.summary import SummarySpider
from scrapy.crawler import CrawlerProcess
from scrapy.signalmanager import dispatcher
from scrapy import signals
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


async def acollect_summarize_urls(urls: list[str], stop: bool = True) -> None:
    """
    Collect and summarize URLs
    """
    results = []

    def crawler_results(signal, sender, item, response, spider):
        results.append(item)

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
        dispatcher.connect(crawler_results, signal=signals.item_scraped)
        print("Starting CrawlerProcess ...")
        process = CrawlerProcess(settings, install_root_handler=False)
        print("Starting crawl ...")
        process.crawl(SummarySpider, urls=urls)
        print("Starting process ...")
        process.start(stop_after_crawl=stop)

    return results


def collect_summarize_urls(urls: list[str], stop: bool = True) -> list[dict]:
    return asyncio.run(acollect_summarize_urls(urls=urls, stop=stop))


print(
    collect_summarize_urls(
        [
            "https://dspy-docs.vercel.app/docs/building-blocks/signatures#inline-dspy-signatures"
        ],
        True,
    )
)
