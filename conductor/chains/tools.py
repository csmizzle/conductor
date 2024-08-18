from serpapi import GoogleSearch


def image_search(query: str, api_key: str) -> dict:
    """Search for images using SerpApi.

    Args:
        query (str): search query
        api_key (str): SerpApi API key

    Returns:
        dict: image search results
    """
    search = GoogleSearch(
        {
            "q": query,
            "api_key": api_key,
        }
    )
    return search.get_dict()
