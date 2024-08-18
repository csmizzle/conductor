from conductor.chains.models import ImageDescription
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class WebPage(BaseModel):
    url: str = Field(..., description="The URL of the webpage")
    created_at: datetime = Field(..., description="The creation date of the webpage")
    content: str = Field(..., description="The content of the webpage")
    raw: str = Field(..., description="The raw content of the webpage")


class SourcedImageDescription(BaseModel):
    created_at: datetime = Field(..., description="The creation date of the image")
    image_description: ImageDescription = Field(
        ..., description="The description of the image"
    )
    source: str = Field(..., description="The source url of the image")
    path: Optional[str] = Field(..., description="The path to the image")
