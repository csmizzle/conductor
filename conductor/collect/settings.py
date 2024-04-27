"""
Scrapy settings module

This module contains Scrapy settings for the project, defining various configurations and options.

For more comprehensive details on Scrapy settings, refer to the official documentation:
http://doc.scrapy.org/en/latest/topics/settings.html
"""

# You can update these options and add new ones
BOT_NAME = "titlebot"
DEPTH_LIMIT = 1
LOG_LEVEL = "INFO"
NEWSPIDER_MODULE = "conductor.collect.spiders"
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
ROBOTSTXT_OBEY = True
SPIDER_MODULES = ["conductor.collect.spiders"]
ITEM_PIPELINES = {
    "conductor.collect.pipelines.TitleItemPipeline": 123,
}
SPIDER_MIDDLEWARES = {
    "conductor.collect.middlewares.TitleSpiderMiddleware": 543,
}
DOWNLOADER_MIDDLEWARES = {
    "conductor.collect.middlewares.TitleDownloaderMiddleware": 543,
}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
