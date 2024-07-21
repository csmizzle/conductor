from pydantic import BaseModel
from datetime import datetime


class WebPage(BaseModel):
    url: str
    created_at: datetime
    content: list[dict]
    raw: str
