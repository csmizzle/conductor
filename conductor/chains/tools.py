from serpapi import GoogleSearch
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
import base64
from conductor.rag.client import zenrows_client
from conductor.chains.models import ImageDescription


image_description_parser = PydanticOutputParser(pydantic_object=ImageDescription)


def image_search(self, query: str, api_key: str) -> dict:
    """Search for images using SerpApi.

    Args:
        query (str): search query
        api_key (str): SerpApi API key

    Returns:
        dict: image search results
    """
    search = GoogleSearch(
        {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
        }
    )
    return search.get_dict()


class ImageProcessor:
    """Simple class to process a single image and describe it using a language model."""

    def __init__(
        self,
        image_content: str,
        model: BaseChatModel,
        metadata: str = None,
    ) -> None:
        self.image_content = image_content
        self.model = model
        self.metadata = metadata
        self.decoded_image = self.process_image()
        self.parser = image_description_parser

    def process_image(self) -> str:
        """Image processing function from https://python.langchain.com/v0.2/docs/how_to/multimodal_inputs/

        Args:
            image_path (str): path to image file

        Returns:
            str: base64 encoded image
        """
        return base64.b64encode(self.image_content).decode("utf-8")

    @classmethod
    def from_image_path(
        cls, image_path: str, model: BaseChatModel, metadata: str = None
    ) -> "ImageProcessor":
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            return cls(image_content=image_data, model=model, metadata=metadata)

    @classmethod
    def from_url(
        cls, image_url: str, model: BaseChatModel, metadata: str = None
    ) -> "ImageProcessor":
        image_content = zenrows_client.get(image_url).content
        return cls(image_content=image_content, model=model, metadata=metadata)

    def describe(self) -> ImageDescription:
        """Describe images from base64 encoded image.

        Args:
            image_base64 (str): base64 encoded image
            model (BaseChatModel): language model to describe the image
            metadata (str, optional): metadata description. Defaults to None.

        Returns:
            BaseMessage: description of the image
        """
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""
                    You are a world class computer vision model.
                    First, evaluate the image and describe the content.
                    Second, evaluate the description and the provided metadata.
                    The metadata will likely include ground truth about the image. Be sure to include this in your description.
                    Finally, return the description of the image combining the description and metadata into a concise summary in the provided JSON format.

                    <metadata>
                    {self.metadata}
                    </metadata>

                    <format_instructions>
                    {self.parser.get_format_instructions()}
                    </format_instructions>
                    """,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.decoded_image}"
                    },
                },
            ]
        )
        response = self.model.invoke([message])
        return self.parser.parse(text=response.content)
