from conductor.chains.tools import image_search
import os


def test_image_search() -> None:
    api_key = os.getenv("SERPAPI_API_KEY")
    images = image_search("apple", api_key)
    assert isinstance(images, dict)
