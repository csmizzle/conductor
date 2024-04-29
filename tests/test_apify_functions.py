"""
Test the apify functions
"""
from conductor.functions.apify_ import run_summarize_urls


def test_run_summarize_urls():
    results = run_summarize_urls(["https://clay.com"])
    assert results is None
