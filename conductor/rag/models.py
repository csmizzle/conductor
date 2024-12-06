from pydantic import BaseModel, Field
from datetime import datetime


class WebPage(BaseModel):
    url: str = Field(..., description="The URL of the webpage")
    created_at: datetime = Field(..., description="The creation date of the webpage")
    content: str = Field(..., description="The content of the webpage")
    raw: str = Field(..., description="The raw content of the webpage")
