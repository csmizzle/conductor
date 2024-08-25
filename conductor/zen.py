from zenrows import ZenRowsClient
import os


# zen rows client
zenrows_client = ZenRowsClient(apikey=os.getenv("ZENROWS_API_KEY"), concurrency=1)


def get_image(url: str) -> str:
    """
    Get an image from a given URL

    Args:
        url (str): URL of the image

    Returns:
        str: The image as a base64 string
    """
    image_response = zenrows_client.get(url)
    if image_response.status_code == 200:
        return image_response.content
