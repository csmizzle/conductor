from pydantic import BaseModel
from datetime import datetime


class WebPage(BaseModel):
    url: str
    created_at: datetime
    title: str
    content: str
