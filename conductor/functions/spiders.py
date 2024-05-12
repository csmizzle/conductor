"""
Conductor spiders module
Implementation of https://stackoverflow.com/questions/41495052/scrapy-reactor-not-restartable
"""
from scrapy.crawler import CrawlerRunner
from scrapy.signalmanager import dispatcher
from scrapy import signals
from multiprocessing import Process, Queue
from twisted.internet import reactor


def f(q, spider, urls: list[str], task_id: str = None):
    try:
        results = []

        def crawler_results(signal, sender, item, response, spider):
            results.append(item)

        dispatcher.connect(crawler_results, signal=signals.item_scraped)
        runner = CrawlerRunner()
        deferred = runner.crawl(spider, urls=urls, task_id=task_id)
        deferred.addBoth(lambda _: reactor.stop())
        reactor.run()
        q.put(results)
    except Exception as e:
        q.put(e)


def run_spider(spider, urls: list[str], task_id: str = None):
    """
    Run Spider in a separate process
    """
    q = Queue()
    p = Process(target=f, args=(q, spider, urls, task_id))
    p.start()
    result = q.get()
    p.join()

    if isinstance(result, Exception):
        raise result
    else:
        return result
