from conductor.chains.tools import image_search, ImageProcessor
from conductor.chains.models import ImageDescription
from conductor.llms import claude_sonnet, openai_gpt_4o
from tests.constants import BASEDIR
import os


def test_image_search() -> None:
    api_key = os.getenv("SERPAPI_API_KEY")
    images = image_search("apple", api_key)
    assert isinstance(images, dict)


def test_describe_image_from_path() -> None:
    image_path = os.path.join(BASEDIR, "data", "test_image.jpg")
    processor = ImageProcessor.from_image_path(
        image_path=image_path,
        model=claude_sonnet,
        metadata="Minimal Logo Design Inspiration: Palantir | DesignRush",
    )
    response = processor.describe()
    assert isinstance(response, ImageDescription)


def test_describe_image_from_url() -> None:
    image_url = "https://media.designrush.com/inspiration_images/135326/conversions/_1511454518_557_palantir1-desktop.jpg"
    processor = ImageProcessor.from_url(
        image_url=image_url,
        model=claude_sonnet,
        metadata="Minimal Logo Design Inspiration: Palantir | DesignRush",
    )
    response = processor.describe()
    assert isinstance(response, ImageDescription)


def test_describe_person_image_from_url() -> None:
    image_url = "https://assets.weforum.org/sf_account/image/-T6sEZZYrPjKBFqgJR9nhnbLpKoafHG__y0ZlbMJaU8.jpg"
    processor = ImageProcessor.from_url(
        image_url=image_url,
        model=openai_gpt_4o,
        metadata="alex karp palantir founder us | Alex Karp | World Economic Forum",
    )
    response = processor.describe()
    assert isinstance(response, ImageDescription)
