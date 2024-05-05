"""
S3 bucket steps
- Get data from discord channel
- Run urls through apify
Similar job id: 1ba67f59-0123-413f-b5b6-ad75b611e0c5-1233762364923445309

"""
from conductor.database.aws import get_object
from conductor.functions.apify_ import run_summarize_urls
import re


def extract_urls(s):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    urls = re.findall(url_pattern, s)
    return urls


def extract_urls_from_object(bucket, key) -> list[str]:
    data = get_object(bucket=bucket, key=key)
    urls = []
    for entry in data:
        urls.extend(extract_urls(entry["message"]))
    return urls


data = get_object(
    bucket="discord-bucket-dev",
    key="1ba67f59-0123-413f-b5b6-ad75b611e0c5-1233762364923445309.json",
)
urls = extract_urls_from_object(
    bucket="discord-bucket-dev",
    key="1ba67f59-0123-413f-b5b6-ad75b611e0c5-1233762364923445309.json",
)
run_summarize_urls(urls=urls)
