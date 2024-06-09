"""
Test sync spider activity
"""
from conductor.functions.crawl import sync_summarize_urls


def test_collect_summarize_urls_without_task_id():
    results = sync_summarize_urls(["https://clay.com"], True)
    assert len(results) > 0


def test_collect_summarize_urls_with_task_id():
    results = sync_summarize_urls(["https://clay.com"], True, task_id="test_task_id")
    assert len(results) > 0
