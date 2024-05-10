"""
Test sync spider activity
"""
from conductor.functions.apify_ import sync_summarize_urls


def test_collect_summarize_urls():
    results = sync_summarize_urls(["https://clay.com"], True)
    assert len(results) > 0
